"""
Symbolic Formula Analysis with Graph Simplification
===================================================

Uses graph_to_formula and expr_to_mini_dag to:
1. Extract symbolic formulas from DAGs
2. Detect equivalent formulas (different graphs, same formula)
3. Build minimal equivalent DAGs
4. Cache formulas to avoid re-evaluation
5. Guide LLM towards simpler, equivalent individuals
"""

from typing import Dict, List, Any, Set, Tuple, Optional
import hashlib
import numpy as np
from sympy import sympify, simplify, Symbol
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class SymbolicAnalysis:
    """Complete symbolic analysis of an individual."""
    original_nodes: int
    minimal_nodes: int
    symbolic_formula: str
    simplified_formula: str
    formula_hash: str
    complexity_reduction: float  # Percentage
    is_equivalent_to: Optional[str]  # Hash of equivalent minimal form
    redundancy_score: float


class SymbolicFormulaAnalyzer:
    """
    Analyzes DAG individuals using symbolic formula extraction.
    
    Uses your graph_to_formula and expr_to_mini_dag functions to:
    - Extract actual mathematical formula from graph
    - Detect equivalent formulas
    - Build minimal DAG
    - Guide optimization
    """
    
    def __init__(self, graph_to_formula_fn, expr_to_mini_dag_fn, input_names: List[str]):
        """
        Parameters
        ----------
        graph_to_formula_fn : callable
            Your graph_to_formula function
        expr_to_mini_dag_fn : callable
            Your expr_to_mini_dag function
        input_names : list[str]
            Input variable names (e.g., ['x0', 'x1'] for NDVI)
        """
        self.graph_to_formula = graph_to_formula_fn
        self.expr_to_mini_dag = expr_to_mini_dag_fn
        self.input_names = input_names
        
        # Cache of symbolic formulas
        self.formula_cache: Dict[str, float] = {}  # formula_hash -> performance
        self.minimal_dags: Dict[str, Any] = {}  # formula_hash -> minimal_dag
        
        # Equivalence classes
        self.equivalence_classes: Dict[str, Set[str]] = defaultdict(set)
    
    def analyze_dag(self, dag_config, X_sample=None) -> SymbolicAnalysis:
        """
        Complete symbolic analysis of a DAG.
        
        Parameters
        ----------
        dag_config : AdjMatrix
            DAG configuration to analyze
        X_sample : array, optional
            Sample input for formula extraction
        
        Returns
        -------
        SymbolicAnalysis
            Complete analysis
        """
        
        # Default sample input if not provided
        X_sample = np.asarray(self.input_names)
        
        try:
            # 1. Extract symbolic formula from DAG
            symbolic_expr = self.graph_to_formula(
                dag_config.matrix,
                X_sample,
                dag_config.operations
            )
            
            # 2. Simplify
            simplified_expr = simplify(symbolic_expr)
            
            # 3. Convert to strings
            symbolic_str = str(symbolic_expr)
            simplified_str = str(simplified_expr)
            
            # 4. Compute hash of simplified formula
            formula_hash = self._compute_formula_hash(simplified_str)
            
            # 5. Build minimal DAG
            try:
                minimal_dag = self.expr_to_mini_dag(simplified_expr, self.input_names)
                minimal_nodes = len(minimal_dag.operations)
                self.minimal_dags[formula_hash] = minimal_dag
            except Exception as e:
                minimal_nodes = len(dag_config.operations)
                print(f"⚠️ Could not build minimal DAG: {e}")
            
            # 6. Calculate metrics
            original_nodes = len(dag_config.operations)
            complexity_reduction = (original_nodes - minimal_nodes) / original_nodes * 100
            redundancy_score = max(0, complexity_reduction / 100)
            
            # 7. Check for equivalence
            equivalent_to = None
            if formula_hash in self.equivalence_classes:
                equivalent_to = formula_hash
            else:
                self.equivalence_classes[formula_hash].add(formula_hash)
            
            return SymbolicAnalysis(
                original_nodes=original_nodes,
                minimal_nodes=minimal_nodes,
                symbolic_formula=symbolic_str,
                simplified_formula=simplified_str,
                formula_hash=formula_hash,
                complexity_reduction=complexity_reduction,
                is_equivalent_to=equivalent_to,
                redundancy_score=redundancy_score
            )
        
        except Exception as e:
            # Fallback if symbolic analysis fails
            print(f"⚠️ Symbolic analysis failed: {e}")
            return SymbolicAnalysis(
                original_nodes=len(dag_config.operations),
                minimal_nodes=len(dag_config.operations),
                symbolic_formula="<extraction failed>",
                simplified_formula="<extraction failed>",
                formula_hash="unknown",
                complexity_reduction=0.0,
                is_equivalent_to=None,
                redundancy_score=0.0
            )
    
    def is_formula_evaluated(self, formula_hash: str) -> Tuple[bool, Optional[float]]:
        """
        Check if this formula was already evaluated.
        
        Returns
        -------
        tuple
            (is_evaluated, cached_performance)
        """
        
        if formula_hash in self.formula_cache:
            return True, self.formula_cache[formula_hash]
        return False, None
    
    def record_evaluation(self, formula_hash: str, performance: float):
        """Record evaluation result for a formula."""
        self.formula_cache[formula_hash] = performance
    
    def get_minimal_dag(self, formula_hash: str) -> Optional[Any]:
        """Get the minimal DAG for a formula."""
        return self.minimal_dags.get(formula_hash)
    
    def get_simplification_suggestions(self, analysis: SymbolicAnalysis) -> List[str]:
        """
        Get suggestions for simplifying individual.
        
        Parameters
        ----------
        analysis : SymbolicAnalysis
            Analysis result
        
        Returns
        -------
        list[str]
            Suggestions
        """
        
        suggestions = []
        
        # High redundancy
        if analysis.complexity_reduction > 20:
            suggestions.append(
                f"⚠️ Architecture can be simplified by {analysis.complexity_reduction:.1f}% "
                f"({analysis.original_nodes} → {analysis.minimal_nodes} nodes)"
            )
            suggestions.append(
                f"💡 Simplified formula: {analysis.simplified_formula}"
            )
        
        # Already evaluated
        if analysis.is_equivalent_to:
            suggestions.append(
                f"⚠️ This formula was already evaluated (equivalent individuals exist)"
            )
        
        # Very complex for simple formula
        if analysis.original_nodes > analysis.minimal_nodes * 1.5:
            suggestions.append(
                f"⚠️ Overcomplex: using {analysis.original_nodes} nodes for formula that needs {analysis.minimal_nodes}"
            )
        
        return suggestions
    
    def _compute_formula_hash(self, formula_str: str) -> str:
        """Compute hash of simplified formula."""
        # Normalize: remove spaces, sort terms if possible
        normalized = formula_str.replace(" ", "")
        return hashlib.md5(normalized.encode()).hexdigest()[:16]


class SymbolicGuidedAnalyzer:
    """
    Enhanced analyzer that uses symbolic analysis to guide the LLM.
    """
    
    def __init__(self, base_analyzer, symbolic_analyzer: SymbolicFormulaAnalyzer):
        self.base_analyzer = base_analyzer
        self.symbolic_analyzer = symbolic_analyzer
    
    def analyze_population(self, state: Dict, save_dir: str, knowledge_base) -> Dict[str, Any]:
        """Analyze with symbolic formula extraction."""
        
        # Standard analysis
        analysis = self.base_analyzer.analyze_population(state, save_dir, knowledge_base)
        
        # Add symbolic analysis for each individual
        import pickle
        
        for profile_dict in analysis['profiles']:
            idx = profile_dict['idx']
            
            try:
                # Load DAG configuration
                with open(f"{save_dir}/x_{idx}.pkl", 'rb') as f:
                    config = pickle.load(f)
                
                # Find DAG in config
                dag = None
                if isinstance(config, list):
                    for item in config:
                        if hasattr(item, 'operations') and hasattr(item, 'matrix'):
                            dag = item
                            break
                else:
                    dag = config
                
                if dag:
                    # Symbolic analysis
                    symbolic = self.symbolic_analyzer.analyze_dag(dag)
                    
                    # Add to profile
                    profile_dict['symbolic_analysis'] = {
                        'formula': symbolic.simplified_formula,
                        'original_nodes': symbolic.original_nodes,
                        'minimal_nodes': symbolic.minimal_nodes,
                        'complexity_reduction': symbolic.complexity_reduction,
                        'redundancy_score': symbolic.redundancy_score
                    }
                    
                    # Record evaluation
                    self.symbolic_analyzer.record_evaluation(
                        symbolic.formula_hash,
                        profile_dict['loss']
                    )
                    
                    # Log interesting cases
                    if symbolic.complexity_reduction > 20:
                        print(f"\n📐 Architecture {idx}: Can be simplified!")
                        print(f"   Formula: {symbolic.simplified_formula}")
                        print(f"   Nodes: {symbolic.original_nodes} → {symbolic.minimal_nodes}")
                        print(f"   Reduction: {symbolic.complexity_reduction:.1f}%")
            
            except Exception as e:
                print(f"⚠️ Could not analyze individual {idx}: {e}")
        
        return analysis


class SymbolicGuidedStrategist:
    """
    Strategist that uses symbolic analysis to make decisions.
    """
    
    def __init__(self, base_strategist, symbolic_analyzer: SymbolicFormulaAnalyzer):
        self.base_strategist = base_strategist
        self.symbolic_analyzer = symbolic_analyzer
    
    def decide_strategy(self, state: Dict, analysis: Dict,
                       allow_train: bool, temperature: float = 0.3) -> Dict:
        """Strategy with symbolic insights."""
        
        # Add symbolic insights to analysis
        symbolic_insights = []
        
        for profile in analysis['profiles']:
            if 'symbolic_analysis' in profile:
                sym = profile['symbolic_analysis']
                if sym['complexity_reduction'] > 20:
                    symbolic_insights.append(
                        f"Architecture {profile['idx']}: {sym['complexity_reduction']:.1f}% redundant "
                        f"(formula: {sym['formula']})"
                    )
        
        if symbolic_insights:
            analysis['symbolic_insights'] = symbolic_insights
            
            print(f"\n📐 SYMBOLIC INSIGHTS:")
            for insight in symbolic_insights[:3]:
                print(f"   • {insight}")
        
        # Standard decision
        return self.base_strategist.decide_strategy(state, analysis, allow_train, temperature)


class SymbolicGuidedArchitect:
    """
    Architect that uses symbolic analysis to design mutations.
    """
    
    def __init__(self, base_architect, symbolic_analyzer: SymbolicFormulaAnalyzer):
        self.base_architect = base_architect
        self.symbolic_analyzer = symbolic_analyzer
    
    def design_mutations(self, target_profile, analysis: Dict, strategy: Dict,
                        temperature: float = 0.4):
        """Design mutations with symbolic guidance."""
        
        # Check if target has symbolic analysis
        if 'symbolic_analysis' in target_profile.__dict__:
            sym = target_profile.symbolic_analysis
            
            # Add symbolic guidance
            analysis['symbolic_guidance'] = {
                'current_formula': sym.get('formula', 'unknown'),
                'complexity_reduction_potential': sym.get('complexity_reduction', 0),
                'minimal_nodes': sym.get('minimal_nodes', target_profile.total_nodes),
                'suggestions': [
                    f"Current formula: {sym.get('formula', 'unknown')}",
                    f"Can reduce from {sym.get('original_nodes', 0)} to {sym.get('minimal_nodes', 0)} nodes"
                ]
            }
            
            print(f"\n📐 SYMBOLIC GUIDANCE:")
            print(f"   Formula: {sym.get('formula', 'unknown')}")
            if sym.get('complexity_reduction', 0) > 10:
                print(f"   ⚠️ Can simplify by {sym['complexity_reduction']:.1f}%")
                print(f"   💡 Target: {sym['minimal_nodes']} nodes")
        
        # Standard mutation design
        return self.base_architect.design_mutations(
            target_profile, analysis, strategy, temperature
        )


def create_symbolic_guided_system(base_policy, graph_to_formula_fn, 
                                  expr_to_mini_dag_fn, input_names: List[str]):
    """
    Enhance multi-agent policy with symbolic formula analysis.
    
    Parameters
    ----------
    base_policy : MultiAgentDAGPolicy
        Base policy to enhance
    graph_to_formula_fn : callable
        Your graph_to_formula function
    expr_to_mini_dag_fn : callable
        Your expr_to_mini_dag function
    input_names : list[str]
        Input variable names (e.g., ['x0', 'x1'])
    
    Returns
    -------
    MultiAgentDAGPolicy
        Enhanced policy with symbolic analysis
    """
    
    # Create symbolic analyzer
    symbolic_analyzer = SymbolicFormulaAnalyzer(
        graph_to_formula_fn,
        expr_to_mini_dag_fn,
        input_names
    )
    
    # Wrap agents
    base_policy.analyzer = SymbolicGuidedAnalyzer(
        base_policy.analyzer,
        symbolic_analyzer
    )
    
    base_policy.strategist = SymbolicGuidedStrategist(
        base_policy.strategist,
        symbolic_analyzer
    )
    
    base_policy.architect = SymbolicGuidedArchitect(
        base_policy.architect,
        symbolic_analyzer
    )
    
    # Store analyzer
    base_policy.symbolic_analyzer = symbolic_analyzer
    
    print("✅ Symbolic formula analysis activated")
    print(f"   Input variables: {input_names}")
    print("   Features:")
    print("     • Extract symbolic formulas from DAGs")
    print("     • Detect equivalent formulas")
    print("     • Build minimal DAGs")
    print("     • Guide towards simpler individuals")
    
    return base_policy
