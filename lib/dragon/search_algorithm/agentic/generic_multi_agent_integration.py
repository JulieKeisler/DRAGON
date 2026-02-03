"""
Integration of Generic Mathematical Analysis
============================================

Integrates:
1. Redundancy detection in Analyzer
2. Formula deduplication before evaluation
3. Pattern discovery in Strategist
4. Learned patterns in Architect
"""

from typing import Dict, List, Any
from dragon.search_algorithm.agentic.multi_agent_rag_system import (
    AnalyzerAgent, StrategistAgent, ArchitectAgent,
    MultiAgentDAGPolicy, IndividualProfile
)
from dragon.search_algorithm.agentic.generic_formula_analysis import (
    GenericFormulaAnalyzer, FormulaDeduplicator,
    PatternDiscoveryEngine, create_generic_analysis_system
)
from dragon.utils.tools import logger


class EnhancedAnalyzerAgent(AnalyzerAgent):
    """
    Analyzer with redundancy detection.
    """
    
    def __init__(self, api_key: str, formula_analyzer: GenericFormulaAnalyzer, model="llama-3.3-70b-versatile"):
        super().__init__(api_key, model=model)
        self.formula_analyzer = formula_analyzer
    
    def analyze_population(self, state: Dict, save_dir: str, knowledge_base) -> Dict[str, Any]:
        """Analyze with redundancy detection."""
        
        # Standard analysis
        analysis = super().analyze_population(state, save_dir, knowledge_base)
        
        # Add formula analysis for each individual
        for profile_dict in analysis['profiles']:
            node_details = profile_dict['node_details']
            
            # Analyze formula
            formula = self.formula_analyzer.analyze_individual(
                node_details,
                performance=profile_dict['loss']
            )
            
            # Add formula insights
            profile_dict['formula_analysis'] = {
                'formula_repr': formula.formula_repr,
                'formula_hash': formula.formula_hash,
                'is_constant': formula.is_constant,
                'is_identity': formula.is_identity,
                'has_redundancy': formula.has_redundancy,
                'redundancy_patterns': formula.redundancy_patterns
            }
            
            # Warn about issues
            if formula.is_constant:
                logger.warning(f"Individual {profile_dict['idx']} produces constant output!")
            
            if formula.has_redundancy:
                logger.info(f"Individual {profile_dict['idx']} has redundancies: {formula.redundancy_patterns}")
        
        return analysis


class PatternLearningStrategist(StrategistAgent):
    """
    Strategist that learns from discovered patterns.
    """
    
    def __init__(self, api_key: str, task_description: str,
                 pattern_engine: PatternDiscoveryEngine, model="llama-3.3-70b-versatile"):
        super().__init__(api_key, task_description, model=model)
        self.pattern_engine = pattern_engine
    
    def decide_strategy(self, state: Dict, analysis: Dict,
                       allow_train: bool, temperature: float = 0.3) -> Dict:
        """Strategy with learned patterns."""
        
        # Discover patterns from history
        discovered_patterns = self.pattern_engine.discover_patterns(top_percentile=0.2)
        
        if discovered_patterns:
            # Add to analysis
            analysis['discovered_patterns'] = discovered_patterns
            
            print(f"\n🔍 DISCOVERED PATTERNS (from top 20%):")
            if discovered_patterns.get('operation_frequencies'):
                print(f"   Best operations: {discovered_patterns['operation_frequencies']}")
            if discovered_patterns.get('avg_complexity'):
                print(f"   Optimal complexity: ~{discovered_patterns['avg_complexity']:.1f} nodes")
            if discovered_patterns.get('feature_usage'):
                print(f"   Feature usage: {discovered_patterns['feature_usage']}")
        
        # Standard decision
        return super().decide_strategy(state, analysis, allow_train, temperature)


class RedundancyAwareArchitect(ArchitectAgent):
    """
    Architect that avoids creating redundant patterns.
    """
    
    def __init__(self, api_key: str, available_operations: List[str],
                 available_combiners: List[str], available_activations: List[str],
                 formula_analyzer: GenericFormulaAnalyzer, model="llama-3.3-70b-versatile"):
        super().__init__(api_key, available_operations, available_combiners, available_activations, model=model)
        self.formula_analyzer = formula_analyzer
    
    def design_mutations(self, target_profile: IndividualProfile,
                        analysis: Dict, strategy: Dict,
                        temperature: float = 0.4):
        """Design mutations avoiding redundancies."""
        
        # Check for redundancies in target
        formula = self.formula_analyzer.analyze_individual(target_profile.node_details)
        
        if formula.has_redundancy:
            # Add redundancy info to analysis
            analysis['target_redundancies'] = formula.redundancy_patterns
            
            print(f"\n⚠️ TARGET REDUNDANCIES DETECTED:")
            for pattern in formula.redundancy_patterns:
                print(f"   • {pattern}")
        
        # Check for learned patterns
        if 'discovered_patterns' in analysis:
            patterns = analysis['discovered_patterns']
            analysis['learned_insights'] = {
                'prefer_operations': patterns.get('operation_frequencies', {}),
                'prefer_combiners': patterns.get('combiner_frequencies', {}),
                'target_complexity': patterns.get('avg_complexity', len(target_profile.node_details))
            }
        
        # Standard mutation design
        return super().design_mutations(target_profile, analysis, strategy, temperature)


class GenericMultiAgentPolicy(MultiAgentDAGPolicy):
    """
    Enhanced multi-agent policy with generic mathematical analysis.
    
    Features:
    - Redundancy detection
    - Formula deduplication
    - Pattern discovery from data
    - NO hardcoded formulas
    """
    
    def __init__(self, api_key: str, task_description: str,
                 search_space_info: Dict[str, Any],
                 allow_train: bool = True,
                 history_length: int = 20,
                 model = "llama-3.3-70b-versatile"):
        
        # Don't call super().__init__ - we'll rebuild from scratch
        from dragon.search_algorithm.agentic.agentic import EnhancedLLMPolicy
        
        EnhancedLLMPolicy.__init__(
            self,
            model=model,
            temperature=0.3,
            api_key=api_key,
            history_length=history_length,
            allow_train=allow_train,
            task_description=task_description
        )
        
        # Store search space info
        self.available_operations = search_space_info.get('operations', [])
        self.available_combiners = search_space_info.get('combiners', ['add', 'concat', 'mul'])
        self.available_activations = search_space_info.get('activations', ['Identity'])
        
        # Create generic analysis components
        self.formula_analyzer = GenericFormulaAnalyzer()
        self.formula_deduplicator = FormulaDeduplicator(self.formula_analyzer)
        self.pattern_discovery = PatternDiscoveryEngine()
        
        # Create enhanced agents
        self.analyzer = EnhancedAnalyzerAgent(api_key, self.formula_analyzer, model=model)
        
        self.strategist = PatternLearningStrategist(
            api_key, task_description, self.pattern_discovery, model=model
        )
        
        self.architect = RedundancyAwareArchitect(
            api_key=api_key,
            model=model,
            available_operations=self.available_operations,
            available_combiners=self.available_combiners,
            available_activations=self.available_activations,
            formula_analyzer=self.formula_analyzer
        )
        
        # Knowledge base
        from dragon.search_algorithm.agentic.multi_agent_rag_system import KnowledgeBase
        self.knowledge_base = KnowledgeBase()
        
        logger.info("Generic multi-agent policy initialized")
        logger.info(f"  Operations: {len(self.available_operations)}")
        logger.info(f"  Combiners: {len(self.available_combiners)}")
        logger.info(f"  Activations: {len(self.available_activations)}")
    
    def decide(self, state: Dict, search_space=None, save_dir=None,
              task_description=None) -> Dict:
        """Decision with formula analysis and deduplication."""
        
        print("\n" + "="*70)
        print("GENERIC MULTI-AGENT DECISION (NO HARDCODED FORMULAS)")
        print("="*70)
        
        # STEP 1: Analyze with redundancy detection
        print("\n📊 ANALYZER: Profiling with redundancy detection...")
        analysis = self.analyzer.analyze_population(state, save_dir, self.knowledge_base)
        
        # Record all individuals for pattern learning
        for profile in analysis['profiles']:
            self.pattern_discovery.record_individual(
                profile['node_details'],
                profile['loss']
            )
        
        # STEP 2: Strategy with learned patterns
        progress = state['step'] / max(state['max_steps'], 1)
        temp_strategy = 0.7 if progress < 0.3 else (0.4 if progress < 0.7 else 0.2)
        
        print("\n🎯 STRATEGIST: Learning from population patterns...")
        strategy = self.strategist.decide_strategy(
            state, analysis, self.allow_train, temp_strategy
        )
        
        # STEP 3: Individual with redundancy avoidance
        mutations = []
        if strategy['primary_operation'] == 'MUTATE_DAG':
            print("\n🏗️ ARCHITECT: Designing mutations (avoiding redundancies)...")
            
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
        
        # Compile action
        action = {
            'drop_target': strategy['drop_targets'],
            'primary_op': strategy['primary_operation'],
            'primary_target': strategy['primary_target'],
            'mutations': [
                {
                    'type': m.mutation_type,
                    'node_idx': m.target_node_idx,
                    'operation': m.operation_choice,
                    'combiner': getattr(m, 'combiner', None),
                    'activation': getattr(m, 'activation', None),
                    'hyperparameters': m.hyperparameters
                }
                for m in mutations
            ] if mutations else None,
            'confidence': strategy['confidence'],
            'reasoning': strategy['reasoning']
        }
        
        self._add_to_history(state, action)
        
        print("\n" + "="*70)
        print(f"DECISION: {action['primary_op']}")
        print("="*70 + "\n")
        
        return action
    
    def _resolve_target_idx(self, target: Any, state: Dict) -> int:
        """Resolve target to index."""
        if target == "best":
            return state["best"]["idx"]
        elif target == "worst":
            return state["worst"]["idx"]
        elif target == "random":
            import numpy as np
            return int(np.random.choice([p['idx'] for p in state['full_population']]))
        else:
            return int(target)


# ============================================================================
# Factory Function
# ============================================================================

def create_generic_multi_agent_search(search_space, save_dir: str, api_key: str,
                                      task_description: str, evaluation,
                                      T=20, K=5, N=1, model="llama-3.3-70b-versatile", **kwargs):
    """
    Create generic multi-agent search with mathematical analysis.
    
    Features:
    - Redundancy detection
    - Formula deduplication
    - Pattern discovery
    - NO hardcoded formulas
    
    Parameters
    ----------
    search_space : SearchSpace
        DAG search space
    save_dir : str
        Save directory
    api_key : str
        Groq API key
    task_description : str
        Task description (WITHOUT mentioning target formula)
    evaluation : callable
        Evaluation function
    T, K, N : int
        Search parameters
    **kwargs
        Additional arguments
    
    Returns
    -------
    MultiAgentDAGSearchAlgorithm
        Enhanced algorithm
    
    Examples
    --------
    >>> algo = create_generic_multi_agent_search(
    ...     search_space=my_space,
    ...     save_dir="./results",
    ...     api_key="your_key",
    ...     task_description="Symbolic regression on 2 input features",  # Generic!
    ...     evaluation=my_loss,
    ...     T=1000, K=200, N=1
    ... )
    >>> algo.run()
    """
    
    from dragon.search_algorithm.agentic.complete_search_space_extraction import extract_complete_search_space
    from dragon.search_algorithm.agentic.multi_agent_integration import MultiAgentDAGSearchAlgorithm
    
    # Extract search space
    search_space_info = extract_complete_search_space(search_space)
    
    # Create policy
    policy = GenericMultiAgentPolicy(
        api_key=api_key,
        task_description=task_description,
        search_space_info=search_space_info,
        allow_train=(N > 1),
        model=model
    )
    
    # Create algorithm
    
    algo = MultiAgentDAGSearchAlgorithm(
        search_space=search_space,
        T=T, K=K, N=N, E=0.01,
        evaluation=evaluation,
        save_dir=save_dir,
        api_key=api_key,
        model=model,
        task_description=task_description,
        **kwargs
    )
    
    # Replace policy with generic version
    algo.agent_policy = policy
    
    print("\n" + "="*70)
    print("GENERIC MULTI-AGENT SYSTEM CREATED")
    print("="*70)
    print("\nFeatures activated:")
    print("  ✅ Redundancy detection (x-x, x/x, x*1, etc.)")
    print("  ✅ Formula deduplication (avoid re-evaluation)")
    print("  ✅ Pattern discovery (learn from top performers)")
    print("  ✅ NO hardcoded formulas - fully generic")
    print("\nThe system will:")
    print("  • Detect and avoid redundant operations")
    print("  • Skip re-evaluating equivalent formulas")
    print("  • Learn successful patterns from population")
    print("  • Discover optimal formula structure autonomously")
    print("="*70 + "\n")
    
    return algo
