"""
Compact Prompts for Multi-Agent System
======================================

Fixes the "max completion tokens reached" error by using shorter prompts.
"""

import json
from dragon.search_algorithm.agentic.multi_agent_rag_system import StrategistAgent
from typing import Dict, Any


class CompactStrategistAgent(StrategistAgent):
    """
    Compact strategist with shorter prompts to avoid token limits.
    """
    
    def _build_strategy_prompt(self, state: Dict, analysis: Dict, allow_train: bool) -> str:
        """Build compact prompt for strategic decision."""
        
        ops = ["MUTATE_DAG", "CREATE"] + (["TRAIN"] if allow_train else [])
        
        # Compact population display (top 5 only)
        best_individuals = sorted(
            analysis['profiles'][:10],
            key=lambda x: x['loss']
        )[:5]
        
        pop_str = "\n".join([
            f"  idx={p['idx']}: loss={p['loss']:.4f}, nodes={p['total_nodes']}"
            for p in best_individuals
        ])
        
        # Compact patterns
        patterns = analysis.get('discovered_patterns', {})
        if patterns and patterns.get('operation_frequencies'):
            # Just show top 3 operations
            ops_freq = patterns['operation_frequencies']
            top_ops = sorted(ops_freq.items(), key=lambda x: x[1], reverse=True)[:3]
            ops_str = ", ".join(
                op.split('.')[-1].replace(">", "").replace("'", "")
                for op, _ in top_ops
            )

        else:
            ops_str = "No patterns yet"
        
        return f"""Strategic decision for evolutionary search: {self.task_description}

STATE (Step {state['step']}/{state['max_steps']} = {state['step']/state['max_steps']*100:.0f}%):
Best: idx={state['best']['idx']}, loss={state['best']['loss']:.4f}
Population: {state['population_size']} individuals

TOP 5:
{pop_str}

PATTERNS (from top 20%):
- Best ops: {ops_str}
- Avg complexity: {patterns.get('avg_complexity', 'N/A')} nodes
- Features used: {patterns.get('feature_usage', {})}

TASK:
1. DROP: 0-2 individuals (worst performers or redundant)
2. PRIMARY: {' | '.join(ops)}
3. TARGET: "best" | "worst" | "random" | <int> | null
4. If MUTATE_DAG: specify 1-3 mutations with node_idx and type

GUIDELINES:
- Early (<30%): Explore (MUTATE_DAG, bold mutations, CREATE if not enough diversity)
- Mid (30-70%): Refine (MUTATE_DAG on promising)
- Late (>70%): Fine-tune (MUTATE_DAG on best, small changes, {f"TRAIN" if allow_train else "focus"})

Return JSON:
{{
    "drop_targets": [<int>] | ["worst"],
    "primary_operation": "{ops[0]}",
    "primary_target": "best" | <int> | null,
    "mutations": [{{"type": "modify"|"add"|"delete", "node_idx": int}}] | null,
    "confidence": 0.0-1.0,
    "reasoning": "<brief explanation>"
}}"""


class CompactArchitectAgent:
    """Compact architect with shorter prompts."""
    
    def __init__(self, api_key: str, available_operations: list,
                 available_combiners: list, available_activations: list):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"
        self.available_operations = available_operations
        self.available_combiners = available_combiners
        self.available_activations = available_activations
    
    def design_mutations(self, target_profile, analysis: Dict, strategy: Dict,
                        temperature: float = 0.4):
        """Design mutations with compact prompt."""
        
        # Compact node display (first 5 nodes only)
        nodes_str = "\n".join([
            f"  {i}: {n['operation']}, {n['combiner']}"
            for i, n in enumerate(target_profile.node_details[:5])
        ])
        
        # Compact operations list
        ops_compact = ", ".join(self.available_operations[:10])
        
        prompt = f"""Design mutations for individuals idx={target_profile.idx} (loss={target_profile.loss:.4f})

CURRENT ({target_profile.total_nodes} nodes):
{nodes_str}
{"  ..." if len(target_profile.node_details) > 5 else ""}

Operations: {len(target_profile.node_details)}
Distribution: {target_profile.operation_distribution}

AVAILABLE:
Ops: {ops_compact}
Combiners: {', '.join(self.available_combiners)}

GOAL: {strategy['reasoning']}

Design 1-3 mutations:
- add: Insert node (specify op, combiner)
- delete: Remove node
- modify: Change op/combiner
- children/parents: Adjust connections

Return JSON:
{{
    "mutations": [
        {{
            "type": "add"|"delete"|"modify"|"children"|"parents",
            "node_idx": int,
            "operation": "OpName" | null,
            "combiner": "add"|"concat"|"mul" | null,
            "rationale": "<why>"
        }}
    ]
}}"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            
            mutations_dict = json.loads(response.choices[0].message.content)
            
            # Parse mutations
            from dragon.search_algorithm.agentic.multi_agent_rag_system import MutationProposal
            mutations = []
            
            for mut in mutations_dict.get('mutations', [])[:3]:
                mutations.append(MutationProposal(
                    mutation_type=mut.get('type', 'modify'),
                    target_node_idx=int(mut.get('node_idx', 0)),
                    operation_choice=mut.get('operation'),
                    hyperparameters=mut.get('hyperparameters'),
                    rationale=mut.get('rationale', '')
                ))
            
            if not mutations:
                mutations = [MutationProposal(
                    mutation_type="modify",
                    target_node_idx=-1,
                    rationale="Default"
                )]
            
            return mutations
            
        except Exception as e:
            from dragon.utils.tools import logger
            logger.error(f"Architect error: {e}")
            
            from dragon.search_algorithm.agentic.multi_agent_rag_system import MutationProposal
            return [MutationProposal(
                mutation_type="modify",
                target_node_idx=-1,
                rationale="Fallback"
            )]


def create_compact_agents(base_policy, api_key, task_desc):
    """
    Replace agents with compact versions to avoid token limits.
    
    Parameters
    ----------
    base_policy : MultiAgentDAGPolicy
        Policy to modify
    
    Returns
    -------
    MultiAgentDAGPolicy
        Modified policy with compact agents
    """
    
    base_policy.strategist = CompactStrategistAgent(
        api_key=api_key,
        task_description=task_desc
    )
    
    # Replace architect
    base_policy.architect = CompactArchitectAgent(
        api_key=api_key,
        available_operations=base_policy.available_operations,
        available_combiners=base_policy.available_combiners,
        available_activations=base_policy.available_activations
    )
    
    print("✅ Compact agents activated (shorter prompts)")
    
    return base_policy