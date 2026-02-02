"""
Industrial Multi-Agent System for DAG  Evolutionary Search
=================================================================

Individual:
    1. Analyzer Agent: Extracts detailed population insights
    2. Strategist Agent: Decides high-level strategy (train/mutate/create/drop)
    3. Architect Agent: Designs specific mutations with operation selection
    
Uses RAG (Retrieval-Augmented Generation) pattern with knowledge base.
"""

import json
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from dragon.utils.tools import logger
from dragon.search_algorithm.agentic.agentic import EnhancedLLMPolicy
from groq import Groq


@dataclass
class IndividualProfile:
    """Detailed profile of an individual."""
    idx: int
    loss: float
    n_evaluations: int
    num_blocks: int
    total_nodes: int
    node_details: List[Dict[str, Any]]
    connectivity_density: float
    operation_distribution: Dict[str, int]
    structural_hash: str
    
    def to_dict(self):
        return asdict(self)


@dataclass
class MutationProposal:
    """Detailed mutation proposal with operation selection."""
    mutation_type: str  # add, delete, modify, children, parents
    target_node_idx: int
    operation_choice: Optional[str] = None  # Specific operation to add/modify to
    hyperparameters: Optional[Dict] = None
    rationale: str = ""


class KnowledgeBase:
    """
    Stores and retrieves architectural patterns and their performance.
    Enables RAG-style querying for similar individuals.
    """
    
    def __init__(self, max_entries=1000):
        self.individuals: List[IndividualProfile] = []
        self.performance_map: Dict[str, List[float]] = {}
        self.mutation_history: List[Dict] = []
        self.max_entries = max_entries
    
    def add_individual(self, profile: IndividualProfile):
        """Add individual to knowledge base."""
        self.individuals.append(profile)
        
        # Track performance by structural hash
        if profile.structural_hash not in self.performance_map:
            self.performance_map[profile.structural_hash] = []
        self.performance_map[profile.structural_hash].append(profile.loss)
        
        # Limit size
        if len(self.individuals) > self.max_entries:
            removed = self.individuals.pop(0)
            # Clean up performance map if needed
    
    def record_mutation(self, parent_idx: int, mutations: List[MutationProposal], 
                       child_idx: int, improvement: float):
        """Record mutation and its outcome."""
        self.mutation_history.append({
            'parent_idx': parent_idx,
            'mutations': [asdict(m) for m in mutations],
            'child_idx': child_idx,
            'improvement': improvement
        })
    
    def get_similar_individuals(self, profile: IndividualProfile, top_k=5) -> List[IndividualProfile]:
        """Find similar individuals (simple similarity for now)."""
        if not self.individuals:
            return []
        
        # Simple similarity: num_blocks and total_nodes
        similarities = []
        for arch in self.individuals:
            score = (
                abs(arch.num_blocks - profile.num_blocks) +
                abs(arch.total_nodes - profile.total_nodes) * 0.1
            )
            similarities.append((score, arch))
        
        # Sort by similarity (lower is better)
        similarities.sort(key=lambda x: x[0])
        return [arch for _, arch in similarities[:top_k]]
    
    def get_successful_mutations(self, top_k=10) -> List[Dict]:
        """Get most successful mutations from history."""
        if not self.mutation_history:
            return []
        
        # Sort by improvement
        sorted_muts = sorted(
            self.mutation_history,
            key=lambda x: x['improvement'],
            reverse=True
        )
        return sorted_muts[:top_k]
    
    def get_operation_statistics(self) -> Dict[str, float]:
        """Get statistics about operation success rates."""
        if not self.individuals:
            return {}
        
        # Calculate average loss per operation type
        op_stats = {}
        for arch in self.individuals:
            for op, count in arch.operation_distribution.items():
                if op not in op_stats:
                    op_stats[op] = []
                op_stats[op].append(arch.loss)
        
        # Average loss per operation
        return {
            op: np.mean(losses) 
            for op, losses in op_stats.items()
        }


class AnalyzerAgent:
    """
    Agent 1: Analyzes population and extracts detailed insights.
    Builds comprehensive architectural profiles.
    """
    
    def __init__(self, api_key: str, model="llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def analyze_population(self, state: Dict, save_dir: str, 
                          knowledge_base: KnowledgeBase) -> Dict[str, Any]:
        """
        Deep analysis of entire population.
        
        Returns
        -------
        dict
            Comprehensive population analysis with:
            - Individual profiles
            - Population-level patterns
            - Recommendations
        """
        
        profiles = []
        
        # Analyze each individual
        for ind in state['full_population']:
            try:
                profile = self._create_individual_profile(
                    ind['idx'], save_dir, ind['loss'], ind['N']
                )
                profiles.append(profile)
                
                # Add to knowledge base
                knowledge_base.add_individual(profile)
                
            except Exception as e:
                logger.warning(f"Could not profile idx {ind['idx']}: {e}")
        
        # Population-level analysis
        analysis = {
            'profiles': [p.to_dict() for p in profiles],
            'population_patterns': self._analyze_patterns(profiles),
            'diversity_metrics': self._compute_diversity(profiles),
            'best_profile': profiles[0].to_dict() if profiles else None,
            'operation_stats': knowledge_base.get_operation_statistics(),
            'successful_mutations': knowledge_base.get_successful_mutations(5)
        }
        
        return analysis
    
    def _create_individual_profile(self, idx: int, save_dir: str, 
                                     loss: float, n_evals: int) -> IndividualProfile:
        """Create detailed profile for one individual."""
        
        with open(f"{save_dir}/x_{idx}.pkl", 'rb') as f:
            config = pickle.load(f)
        
        # Extract DAG from config
        dag = None
        for item in config:
            if hasattr(item, 'operations') and hasattr(item, 'matrix'):
                dag = item
                break
        
        if dag is None:
            raise ValueError(f"No DAG found in config {idx}")
        
        # Extract detailed node information
        node_details = []
        for i, node in enumerate(dag.operations):
            node_info = {
                'index': i,
                'combiner': getattr(node, 'combiner', 'unknown'),
                'operation': str(node.name) if hasattr(node, 'name') else 'unknown',
                'activation': str(node.activation) if hasattr(node, 'activation') else 'unknown',
                'hyperparameters': getattr(node, 'hp', {})
            }
            node_details.append(node_info)
        
        # Compute connectivity density
        total_possible = len(dag.operations) * (len(dag.operations) - 1) / 2
        total_actual = np.sum(dag.matrix) if hasattr(dag, 'matrix') else 0
        connectivity_density = total_actual / total_possible if total_possible > 0 else 0
        
        # Operation distribution
        op_dist = {}
        for node in node_details:
            op = node['operation']
            op_dist[op] = op_dist.get(op, 0) + 1
        
        # Structural hash (simple version)
        structural_hash = f"{len(dag.operations)}_{len(op_dist)}_{connectivity_density:.2f}"
        
        return IndividualProfile(
            idx=idx,
            loss=float(loss),
            n_evaluations=int(n_evals),
            num_blocks=1,  # Simplified for now
            total_nodes=len(dag.operations),
            node_details=node_details,
            connectivity_density=float(connectivity_density),
            operation_distribution=op_dist,
            structural_hash=structural_hash
        )
    
    def _analyze_patterns(self, profiles: List[IndividualProfile]) -> Dict:
        """Identify architectural patterns."""
        if not profiles:
            return {}
        
        # Best performers
        sorted_profiles = sorted(profiles, key=lambda p: p.loss)
        best = sorted_profiles[0]
        worst = sorted_profiles[-1]
        
        # Common operations in top performers
        top_3 = sorted_profiles[:3]
        all_ops = {}
        for p in top_3:
            for op, count in p.operation_distribution.items():
                all_ops[op] = all_ops.get(op, 0) + count
        
        return {
            'best_individual': {
                'idx': best.idx,
                'loss': best.loss,
                'nodes': best.total_nodes,
                'operations': best.operation_distribution
            },
            'worst_individual': {
                'idx': worst.idx,
                'loss': worst.loss
            },
            'common_operations_in_top': all_ops,
            'avg_nodes': np.mean([p.total_nodes for p in profiles]),
            'avg_connectivity': np.mean([p.connectivity_density for p in profiles])
        }
    
    def _compute_diversity(self, profiles: List[IndividualProfile]) -> Dict:
        """Compute population diversity metrics."""
        if len(profiles) < 2:
            return {'structural_diversity': 0}
        
        # Unique structural hashes
        unique_structures = len(set(p.structural_hash for p in profiles))
        
        # Variance in number of nodes
        node_counts = [p.total_nodes for p in profiles]
        node_variance = np.var(node_counts) if node_counts else 0
        
        return {
            'structural_diversity': unique_structures / len(profiles),
            'node_count_variance': float(node_variance),
            'unique_structures': unique_structures,
            'total_population': len(profiles)
        }


class StrategistAgent:
    """
    Agent 2: Makes high-level strategic decisions.
    Decides: DROP which individuals, and primary operation (TRAIN/MUTATE_DAG/CREATE).
    """
    
    def __init__(self, api_key: str, task_description: str, model="llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.task_description = task_description
    
    def decide_strategy(self, state: Dict, analysis: Dict, 
                       allow_train: bool, temperature: float = 0.3) -> Dict:
        """
        High-level strategic decision.
        
        Returns
        -------
        dict
            {
                'drop_targets': list[int],
                'primary_operation': 'TRAIN' | 'MUTATE_DAG' | 'CREATE',
                'primary_target': 'best' | 'worst' | int,
                'confidence': float,
                'reasoning': str
            }
        """
        
        prompt = self._build_strategy_prompt(state, analysis, allow_train)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            
            decision = json.loads(response.choices[0].message.content)
            
            # Validate and fix
            decision = self._validate_strategy(decision, state, allow_train)
            
            print(f"\n🎯 STRATEGIST: {decision['primary_operation']} on {decision['primary_target']}")
            print(f"   Drop: {decision['drop_targets']}, Confidence: {decision['confidence']:.2f}")
            print(f"   Reasoning: {decision['reasoning']}")
            
            return decision
            
        except Exception as e:
            logger.error(f"Strategist error: {e}")
            return {
                'drop_targets': ['worst'],
                'primary_operation': 'CREATE',
                'primary_target': None,
                'confidence': 0.3,
                'reasoning': 'Fallback decision'
            }
    
    def _build_strategy_prompt(self, state: Dict, analysis: Dict, allow_train: bool) -> str:
        """Build prompt for strategic decision."""
        
        ops = ["MUTATE_DAG", "CREATE"] + (["TRAIN"] if allow_train else [])
        
        return f"""You are a STRATEGIST agent for a graph-encoding evolutionary search: {self.task_description}.

CURRENT STATE (Step {state['step']}/{state['max_steps']}, {state['step']/state['max_steps']*100:.0f}%):
- Population: {state['population_size']} individuals
- Best: idx={state['best']['idx']}, loss={state['best']['loss']:.6f}, N={state['best']['N']}
- Worst: idx={state['worst']['idx']}, loss={state['worst']['loss']:.6f}
- Stats: μ={state['stats']['mean_loss']:.6f}, σ={state['stats']['std_loss']:.6f}

POPULATION ANALYSIS:
{json.dumps(analysis['population_patterns'], indent=2)}

DIVERSITY:
{json.dumps(analysis['diversity_metrics'], indent=2)}

TOP PERFORMING OPERATIONS:
{json.dumps(analysis.get('operation_stats', {}), indent=2)[:500]}

SUCCESSFUL PAST MUTATIONS:
{json.dumps(analysis.get('successful_mutations', [])[:2], indent=2)[:500]}

YOUR TASK:
Make HIGH-LEVEL strategic decisions:

1. DROP STRATEGY:
   - Identify 1-3 individuals to remove
   - Prioritize: high loss, structural redundancy, failed patterns
   
2. PRIMARY OPERATION:
   - TRAIN: Continue training promising individual
   - MUTATE_DAG: Evolve individual (will be detailed by Architect agent)
   - CREATE: Generate new random individual

3. PRIMARY TARGET:
   - Which individual to operate on
   - Can be "best", "worst", "random", or specific index

GUIDELINES:
- Early (<30%): Explore (prefer MUTATE_DAG, diverse targets, CREATE if not enough diversity){f", TRAIN diverse targets" if allow_train else ""}
- Mid (30-70%): Exploit patterns (MUTATE_DAG {f" or TRAIN" if allow_train else ""} on promising individuals) 
- Late (>70%): Refine ({f"TRAIN best, " if allow_train else ""}careful MUTATE_DAG)
- Low diversity: Increase exploration
- Stagnation: Try different approach

Return ONLY JSON:
{{
    "drop_targets": [<int>, <int>] | ["worst"] | ["best", "worst"],
    "primary_operation": "{'" | "'.join(ops)}",
    "primary_target": "best" | "worst" | "random" | <int> | null,
    "confidence": <float 0-1>,
    "reasoning": "<strategic rationale>"
}}"""
    
    def _validate_strategy(self, decision: Dict, state: Dict, allow_train: bool) -> Dict:
        """Validate and fix strategic decision."""
        
        # Ensure drop_targets is list
        if not isinstance(decision.get('drop_targets'), list):
            decision['drop_targets'] = [decision.get('drop_targets', 'worst')]
        
        # Validate operation
        ops = ["MUTATE_DAG", "CREATE"] + (["TRAIN"] if allow_train else [])
        if decision.get('primary_operation') not in ops:
            decision['primary_operation'] = "CREATE"
        
        # Validate target
        if decision['primary_operation'] != "CREATE":
            pt = decision.get('primary_target')
            if pt not in ["best", "worst", "random"] and not isinstance(pt, int):
                decision['primary_target'] = "best"
        
        # Validate confidence
        decision['confidence'] = max(0.0, min(1.0, float(decision.get('confidence', 0.5))))
        
        if 'reasoning' not in decision:
            decision['reasoning'] = "No reasoning"
        
        return decision


class ArchitectAgent:
    """
    Agent 3: Designs detailed mutations.
    Given a strategy to MUTATE_DAG, specifies exactly:
    - Which nodes to modify/add/delete
    - Which operations to use
    - Which combiners to use
    - Which activations to use
    - Hyperparameters
    """
    
    def __init__(self, api_key: str, available_operations: List[str],
                 available_combiners: List[str], available_activations: List[str], model="llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.available_operations = available_operations
        self.available_combiners = available_combiners
        self.available_activations = available_activations
    
    def design_mutations(self, target_profile: IndividualProfile,
                        analysis: Dict, strategy: Dict,
                        temperature: float = 0.4) -> List[MutationProposal]:
        """
        Design specific mutations for target individual.
        
        Returns
        -------
        list[MutationProposal]
            Detailed mutation specifications
        """
        
        prompt = self._build_architect_prompt(target_profile, analysis, strategy)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            
            mutations_dict = json.loads(response.choices[0].message.content)
            mutations = self._parse_mutations(mutations_dict)
            
            print(f"\n🏗️ ARCHITECT: Designed {len(mutations)} mutations")
            for mut in mutations:
                print(f"   - {mut.mutation_type} @ node {mut.target_node_idx}: {mut.rationale}")
            
            return mutations
            
        except Exception as e:
            logger.error(f"Architect error: {e}")
            # Fallback: simple modify
            return [MutationProposal(
                mutation_type="modify",
                target_node_idx=-1,
                rationale="Fallback mutation"
            )]
    
    def _build_architect_prompt(self, profile: IndividualProfile,
                                analysis: Dict, strategy: Dict) -> str:
        """Build detailed prompt for mutation design."""
        
        return f"""You are an ARCHITECT agent designing specific DAG mutations.

TARGET INDIVIDUAL (idx={profile.idx}, loss={profile.loss:.6f}):
Total nodes: {profile.total_nodes}
Connectivity: {profile.connectivity_density:.2f}

NODE DETAILS:
{json.dumps(profile.node_details[:5], indent=2)}

OPERATION DISTRIBUTION:
{json.dumps(profile.operation_distribution, indent=2)}

POPULATION CONTEXT:
Best operations: {analysis.get('population_patterns', {}).get('common_operations_in_top', {})}
Average nodes: {analysis.get('population_patterns', {}).get('avg_nodes', 0)}

STRATEGIC GOAL:
{strategy['reasoning']}

AVAILABLE SEARCH SPACE:

Operations ({len(self.available_operations)}):
{', '.join(self.available_operations)}

Combiners ({len(self.available_combiners)}):
{', '.join(self.available_combiners)}

Activations ({len(self.available_activations)}):
{', '.join(self.available_activations)}

YOUR TASK:
Design 1-3 specific mutations to improve this individual.

MUTATION TYPES:
- add: Insert new node (specify operation, combiner, activation)
- delete: Remove node (specify which)
- modify: Change node (specify new operation/combiner/activation)
- children: Adjust output connections
- parents: Adjust input connections

GUIDELINES:
- Consider node interactions
- change connections to modify the information flow
- remove useless connections
- Preserve successful patterns from analysis
- Choose operations/combiners/activations based on population success
- Each mutation should have clear purpose

Return ONLY JSON:
{{
    "mutations": [
        {{
            "type": "add" | "delete" | "modify" | "children" | "parents",
            "node_idx": <int>,
            "operation": "<operation name>" | null,
            "combiner": "<combiner>" | null,
            "activation": "<activation>" | null,
            "hyperparameters": {{"key": value}} | null,
            "rationale": "<why this mutation>"
        }}
    ]
}}

Example mutation (add):
{{
    "type": "add",
    "node_idx": 2,
    "operation": "Attention1D",
    "combiner": "concat",
    "activation": "GELU",
    "hyperparameters": {{"heads": 4}},
    "rationale": "Add attention with concat for richer feature combination"
}}

Example mutation (modify):
{{
    "type": "modify",
    "node_idx": 3,
    "operation": "SumFeatures",
    "combiner": "add",
    "activation": "Identity",
    "hyperparameters": null,
    "rationale": "Change to SumFeatures with add combiner for better aggregation"
}}"""
    
    def _parse_mutations(self, mutations_dict: Dict) -> List[MutationProposal]:
        """Parse LLM response into MutationProposal objects."""
        
        proposals = []
        
        for mut in mutations_dict.get('mutations', [])[:3]:
            # Validate operation choice
            op_choice = mut.get('operation')
            if op_choice and op_choice not in self.available_operations:
                # Try to find closest match
                op_choice = self._find_closest_operation(op_choice)
            
            proposal = MutationProposal(
                mutation_type=mut.get('type', 'modify'),
                target_node_idx=int(mut.get('node_idx', 0)),
                operation_choice=op_choice,
                hyperparameters=mut.get('hyperparameters'),
                rationale=mut.get('rationale', '')
            )
            proposals.append(proposal)
        
        return proposals if proposals else [MutationProposal(
            mutation_type="modify",
            target_node_idx=0,
            rationale="Default mutation"
        )]
    
    def _find_closest_operation(self, requested: str) -> Optional[str]:
        """Find closest matching operation."""
        requested_lower = requested.lower()
        for op in self.available_operations:
            if requested_lower in op.lower() or op.lower() in requested_lower:
                return op
        return None


class MultiAgentDAGPolicy(EnhancedLLMPolicy):
    """
    Orchestrates multiple agents for industrial-grade DAG evolution.
    
    Uses RAG pattern with knowledge base.
    NOW includes complete search space: operations, combiners, activations.
    """
    
    def __init__(self, api_key: str, task_description: str,
                 search_space_info: Dict[str, Any],
                 allow_train: bool = True,
                 history_length: int = 20,
                 model="llama-3.3-70b-versatile"):
        
        super().__init__(
            model=model,
            temperature=0.3,
            api_key=api_key,
            history_length=history_length,
            allow_train=allow_train,
            task_description=task_description
        )
        
        # Store complete search space info
        self.available_operations = search_space_info.get('operations', [])
        self.available_combiners = search_space_info.get('combiners', ['add', 'concat', 'mul'])
        self.available_activations = search_space_info.get('activations', ['Identity'])
        self.operation_details = search_space_info.get('operation_details', {})
        self.hyperparameter_spaces = search_space_info.get('hyperparameter_spaces', {})
        
        # Initialize agents with complete info
        self.analyzer = AnalyzerAgent(api_key, model=model)
        self.strategist = StrategistAgent(api_key, task_description, model=model)
        self.architect = ArchitectAgent(
            api_key=api_key,
            available_operations=self.available_operations,
            available_combiners=self.available_combiners,
            available_activations=self.available_activations,
            model=model
        )
        
        # Knowledge base for RAG
        self.knowledge_base = KnowledgeBase()
    
    def decide(self, state: Dict, search_space=None, save_dir=None, 
              task_description=None) -> Dict:
        """
        Multi-agent decision process.
        
        Flow:
        1. Analyzer: Deep population analysis
        2. Strategist: High-level decisions (drop, operation, target)
        3. Architect: Detailed mutations (if MUTATE_DAG)
        """
        
        print("\n" + "="*60)
        print("MULTI-AGENT DECISION PROCESS")
        print("="*60)
        
        # STEP 1: Analysis
        print("\n📊 ANALYZER: Profiling population...")
        analysis = self.analyzer.analyze_population(state, save_dir, self.knowledge_base)
        
        # STEP 2: Strategy
        progress = state['step'] / max(state['max_steps'], 1)
        temp_strategy = 0.7 if progress < 0.3 else (0.4 if progress < 0.7 else 0.2)
        
        print("\n🎯 STRATEGIST: Making strategic decision...")
        strategy = self.strategist.decide_strategy(
            state, analysis, self.allow_train, temp_strategy
        )
        
        # STEP 3: Individual (if mutating)
        mutations = []
        if strategy['primary_operation'] == 'MUTATE_DAG':
            print("\n🏗️ ARCHITECT: Designing mutations...")
            
            # Get target profile
            target_idx = self._resolve_target_idx(strategy['primary_target'], state)
            target_profile = next(
                (IndividualProfile(**p) for p in analysis['profiles'] if p['idx'] == target_idx),
                None
            )
            
            if target_profile:
                temp_architect = 0.5 if progress < 0.5 else 0.3
                mutations = self.architect.design_mutations(
                    target_profile, analysis, strategy, temp_architect
                )
        
        # Compile final action
        action = {
            'drop_target': strategy['drop_targets'],
            'primary_op': strategy['primary_operation'],
            'primary_target': strategy['primary_target'],
            'mutations': [
                {
                    'type': m.mutation_type,
                    'node_idx': m.target_node_idx,
                    'operation': m.operation_choice,
                    'hyperparameters': m.hyperparameters
                }
                for m in mutations
            ] if mutations else None,
            'confidence': strategy['confidence'],
            'reasoning': strategy['reasoning'] + (
                f" | Mutations: {[m.rationale for m in mutations]}" if mutations else ""
            )
        }
        
        self._add_to_history(state, action)
        
        print("\n" + "="*60)
        print(f"FINAL DECISION: {action['primary_op']}")
        print("="*60 + "\n")
        
        return action
    
    def _resolve_target_idx(self, target: Any, state: Dict) -> int:
        """Resolve target to actual index."""
        if target == "best":
            return state["best"]["idx"]
        elif target == "worst":
            return state["worst"]["idx"]
        elif target == "random":
            return int(np.random.choice([p['idx'] for p in state['full_population']]))
        else:
            return int(target)