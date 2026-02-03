"""
Generic Mathematical Pattern Discovery System
==============================================

Discovers patterns and redundancies WITHOUT prior knowledge of the target formula.

Key Features:
1. Detects redundant patterns (x-x, x/x, constants)
2. Extracts symbolic formulas from DAGs
3. Caches evaluated formulas to avoid re-evaluation
4. Discovers successful patterns from population history
5. NO hardcoded target formulas - fully generic
"""

from typing import Dict, List, Any, Set, Tuple, Optional
from dataclasses import dataclass
import hashlib
import json
from collections import defaultdict
import numpy as np


@dataclass
class SymbolicFormula:
    """Symbolic representation of a DAG."""
    formula_hash: str
    formula_repr: str
    node_count: int
    operation_sequence: List[str]
    is_constant: bool
    is_identity: bool
    has_redundancy: bool
    redundancy_patterns: List[str]


class GenericFormulaAnalyzer:
    """
    Generic analyzer that discovers mathematical patterns without prior knowledge.
    
    Detects:
    - Redundant operations (x-x, x/x, x*1, x+0)
    - Constant outputs
    - Identity mappings
    - Repeated subgraphs
    """
    
    def __init__(self):
        # Cache of evaluated formulas
        self.formula_cache: Dict[str, float] = {}
        
        # Pattern discovery
        self.successful_patterns: Dict[str, List[float]] = defaultdict(list)
        self.failed_patterns: Set[str] = set()
        
        # Redundancy detection
        self.redundancy_rules = [
            self._detect_subtract_self,
            self._detect_divide_self,
            self._detect_multiply_one,
            self._detect_add_zero,
            self._detect_inverse_inverse,
            self._detect_unused_features,
            self._detect_repeated_operations
        ]
    
    def analyze_individual(self, node_details: List[Dict], 
                            performance: Optional[float] = None) -> SymbolicFormula:
        """
        Analyze individual to extract symbolic formula.
        
        Parameters
        ----------
        node_details : list[dict]
            Nodes with operation, combiner, hyperparameters
        performance : float, optional
            Loss/score of this individual
            
        Returns
        -------
        SymbolicFormula
            Symbolic analysis
        """
        
        # Extract symbolic formula
        formula_repr = self._extract_symbolic_formula(node_details)
        formula_hash = self._compute_formula_hash(formula_repr, node_details)
        
        # Detect redundancies
        redundancies = []
        for rule in self.redundancy_rules:
            pattern = rule(node_details)
            if pattern:
                redundancies.append(pattern)
        
        # Check if constant or identity
        is_constant = self._is_constant_output(node_details)
        is_identity = self._is_identity_mapping(node_details)
        
        # Build operation sequence
        op_sequence = [n.get('operation', 'Unknown') for n in node_details]
        
        # Record performance if provided
        if performance is not None:
            self.formula_cache[formula_hash] = performance
            
            # Learn patterns
            pattern_key = self._extract_pattern_signature(node_details)
            self.successful_patterns[pattern_key].append(performance)
        
        return SymbolicFormula(
            formula_hash=formula_hash,
            formula_repr=formula_repr,
            node_count=len(node_details),
            operation_sequence=op_sequence,
            is_constant=is_constant,
            is_identity=is_identity,
            has_redundancy=len(redundancies) > 0,
            redundancy_patterns=redundancies
        )
    
    def is_formula_cached(self, formula_hash: str) -> bool:
        """Check if this formula was already evaluated."""
        return formula_hash in self.formula_cache
    
    def get_cached_performance(self, formula_hash: str) -> Optional[float]:
        """Get cached performance for formula."""
        return self.formula_cache.get(formula_hash)
    
    def discover_successful_patterns(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Discover which patterns lead to good performance.
        
        Returns
        -------
        list[tuple]
            (pattern_signature, avg_performance) sorted by performance
        """
        
        pattern_performance = []
        
        for pattern, performances in self.successful_patterns.items():
            if len(performances) >= 2:  # At least 2 samples
                avg_perf = np.mean(performances)
                pattern_performance.append((pattern, avg_perf))
        
        # Sort by performance (assuming lower is better)
        pattern_performance.sort(key=lambda x: x[1])
        
        return pattern_performance[:top_k]
    
    # ========================================================================
    # SYMBOLIC FORMULA EXTRACTION
    # ========================================================================
    
    def _extract_symbolic_formula(self, nodes: List[Dict]) -> str:
        """
        Extract human-readable symbolic formula.
        
        Examples:
        - "Select[0] + Select[1]"
        - "Inverse(Select[0] + Select[1])"
        - "Select[0] * Inverse(Select[1])"
        """
        
        # Build dependency graph
        formula_parts = []
        
        for i, node in enumerate(nodes):
            op = node.get('operation', 'Unknown')
            combiner = node.get('combiner', 'add')
            hp = node.get('hyperparameters', {})
            
            # Simplify operation names
            if 'SelectFeatures' in op:
                feat_idx = hp.get('feature_indices', [0])[0]
                formula_parts.append(f"F{feat_idx}")
            elif 'SumFeatures' in op:
                formula_parts.append("Sum")
            elif 'Inverse' in op:
                formula_parts.append("Inv")
            elif 'Negate' in op:
                formula_parts.append("Neg")
            elif 'Identity' in op:
                formula_parts.append("Id")
            else:
                formula_parts.append(op[:3])
        
        # Combine with combiners
        formula = " → ".join(formula_parts)
        return formula
    
    def _compute_formula_hash(self, formula_repr: str, nodes: List[Dict]) -> str:
        """Compute unique hash for formula structure."""
        
        # Include operation sequence and combiners
        structure = []
        for node in nodes:
            op = node.get('operation', '')
            combiner = node.get('combiner', '')
            hp = node.get('hyperparameters', {})
            structure.append(f"{op}:{combiner}:{sorted(hp.items())}")
        
        structure_str = "|".join(structure)
        return hashlib.md5(structure_str.encode()).hexdigest()[:16]
    
    def _extract_pattern_signature(self, nodes: List[Dict]) -> str:
        """
        Extract high-level pattern signature.
        
        Example: "Select(2)→Sum(1)→Inverse(2)→Identity(3)"
        """
        
        op_counts = defaultdict(int)
        for node in nodes:
            op = node.get('operation', 'Unknown')
            # Simplify name
            if 'Select' in op:
                op = 'Select'
            elif 'Sum' in op:
                op = 'Sum'
            elif 'Inverse' in op:
                op = 'Inverse'
            elif 'Identity' in op:
                op = 'Identity'
            op_counts[op] += 1
        
        # Sort by count
        signature_parts = [f"{op}({cnt})" for op, cnt in sorted(op_counts.items())]
        return "→".join(signature_parts)
    
    # ========================================================================
    # REDUNDANCY DETECTION
    # ========================================================================
    
    def _detect_subtract_self(self, nodes: List[Dict]) -> Optional[str]:
        """Detect x - x patterns (always 0)."""
        
        # Look for pattern: same feature selected, then negated/inversed
        select_nodes = [(i, n) for i, n in enumerate(nodes) if 'SelectFeatures' in n.get('operation', '')]
        
        if len(select_nodes) < 2:
            return None
        
        # Check if selecting same feature multiple times with inverse/negate
        features_selected = {}
        for idx, node in select_nodes:
            hp = node.get('hyperparameters', {})
            feat_idx = tuple(hp.get('feature_indices', []))
            if feat_idx in features_selected:
                # Same feature selected twice - check for cancellation
                return f"REDUNDANCY: Feature {feat_idx} selected multiple times (potential x-x)"
            features_selected[feat_idx] = idx
        
        return None
    
    def _detect_divide_self(self, nodes: List[Dict]) -> Optional[str]:
        """Detect x / x patterns (always 1)."""
        
        # Look for inverse operations that cancel out
        inverse_count = sum(1 for n in nodes if 'Inverse' in n.get('operation', ''))
        
        # If multiple inverses with mul combiner, might be x/x
        if inverse_count >= 2:
            mul_combiners = sum(1 for n in nodes if n.get('combiner') == 'mul')
            if mul_combiners >= 2:
                return "POTENTIAL: Multiple inverses with mul (check for x/x=1)"
        
        return None
    
    def _detect_multiply_one(self, nodes: List[Dict]) -> Optional[str]:
        """Detect x * 1 patterns (identity)."""
        
        # Count identity operations with mul combiner
        identity_with_mul = sum(
            1 for n in nodes 
            if 'Identity' in n.get('operation', '') and n.get('combiner') == 'mul'
        )
        
        if identity_with_mul > 0:
            return f"REDUNDANCY: {identity_with_mul}x Identity with mul combiner (x*1=x)"
        
        return None
    
    def _detect_add_zero(self, nodes: List[Dict]) -> Optional[str]:
        """Detect x + 0 patterns."""
        
        # Excessive identity with add combiner
        identity_with_add = sum(
            1 for n in nodes 
            if 'Identity' in n.get('operation', '') and n.get('combiner') == 'add'
        )
        
        if identity_with_add > 3:
            return f"REDUNDANCY: {identity_with_add}x Identity with add (potentially x+0)"
        
        return None
    
    def _detect_inverse_inverse(self, nodes: List[Dict]) -> Optional[str]:
        """Detect Inv(Inv(x)) = x patterns."""
        
        # Look for consecutive inverse operations
        for i in range(len(nodes) - 1):
            if 'Inverse' in nodes[i].get('operation', ''):
                if 'Inverse' in nodes[i+1].get('operation', ''):
                    return "REDUNDANCY: Consecutive inverse operations (Inv(Inv(x))=x)"
        
        return None
    
    def _detect_unused_features(self, nodes: List[Dict]) -> Optional[str]:
        """Detect if some input features are never used."""
        
        # Count which features are selected
        features_used = set()
        for node in nodes:
            if 'SelectFeatures' in node.get('operation', ''):
                hp = node.get('hyperparameters', {})
                feat_indices = hp.get('feature_indices', [])
                features_used.update(feat_indices)
        
        # If only 1 feature used, might be missing information
        if len(features_used) == 1:
            return f"WARNING: Only feature {list(features_used)} used (other features ignored)"
        
        return None
    
    def _detect_repeated_operations(self, nodes: List[Dict]) -> Optional[str]:
        """Detect unnecessarily repeated operations."""
        
        # Count operation types
        op_counts = defaultdict(int)
        for node in nodes:
            op = node.get('operation', '')
            hp = node.get('hyperparameters', {})
            # Include hyperparameters in key for exact matching
            op_key = f"{op}:{sorted(hp.items())}"
            op_counts[op_key] += 1
        
        # Check for excessive repeats
        redundant = []
        for op_key, count in op_counts.items():
            if count > 2 and 'Identity' not in op_key:
                op_name = op_key.split(':')[0]
                redundant.append(f"{count}x {op_name}")
        
        if redundant:
            return f"REDUNDANCY: Repeated operations - {', '.join(redundant)}"
        
        return None
    
    def _is_constant_output(self, nodes: List[Dict]) -> bool:
        """Check if individual likely produces constant output."""
        
        # No feature selection = constant
        has_select = any('SelectFeatures' in n.get('operation', '') for n in nodes)
        
        if not has_select:
            return True
        
        # All inverses cancel out = might be constant
        inverse_count = sum(1 for n in nodes if 'Inverse' in n.get('operation', ''))
        if inverse_count >= 4:  # Too many cancellations
            return True
        
        return False
    
    def _is_identity_mapping(self, nodes: List[Dict]) -> bool:
        """Check if individual is just identity (output = input)."""
        
        # Only identity operations
        non_identity = [n for n in nodes if 'Identity' not in n.get('operation', '')]
        
        return len(non_identity) == 0


class FormulaDeduplicator:
    """
    Deduplicates formulas to avoid re-evaluating equivalent individuals.
    """
    
    def __init__(self, analyzer: GenericFormulaAnalyzer):
        self.analyzer = analyzer
        self.seen_formulas: Set[str] = set()
    
    def is_duplicate(self, node_details: List[Dict]) -> Tuple[bool, Optional[float]]:
        """
        Check if this formula was already evaluated.
        
        Returns
        -------
        tuple
            (is_duplicate, cached_performance)
        """
        
        formula = self.analyzer.analyze_individual(node_details)
        
        if formula.formula_hash in self.seen_formulas:
            cached_perf = self.analyzer.get_cached_performance(formula.formula_hash)
            return True, cached_perf
        
        self.seen_formulas.add(formula.formula_hash)
        return False, None
    
    def should_skip_evaluation(self, node_details: List[Dict]) -> Tuple[bool, str]:
        """
        Determine if evaluation should be skipped.
        
        Returns
        -------
        tuple
            (should_skip, reason)
        """
        
        formula = self.analyzer.analyze_individual(node_details)
        
        # Skip constants
        if formula.is_constant:
            return True, "Produces constant output"
        
        # Skip identity
        if formula.is_identity:
            return True, "Pure identity mapping"
        
        # Skip if already evaluated
        if self.analyzer.is_formula_cached(formula.formula_hash):
            cached = self.analyzer.get_cached_performance(formula.formula_hash)
            return True, f"Already evaluated (loss={cached:.4f})"
        
        return False, ""


class PatternDiscoveryEngine:
    """
    Discovers successful patterns from population history.
    
    NO hardcoded knowledge - learns purely from data.
    """
    
    def __init__(self):
        self.performance_history: List[Dict] = []
    
    def record_individual(self, node_details: List[Dict], performance: float):
        """Record an evaluated individual."""
        
        self.performance_history.append({
            'nodes': node_details,
            'performance': performance,
            'complexity': len(node_details)
        })
    
    def discover_patterns(self, top_percentile: float = 0.1) -> Dict[str, Any]:
        """
        Discover what makes good individuals.
        
        Parameters
        ----------
        top_percentile : float
            Top % of individuals to analyze
        
        Returns
        -------
        dict
            Discovered patterns
        """
        
        if not self.performance_history:
            return {}
        
        # Sort by performance (assuming lower is better)
        sorted_history = sorted(self.performance_history, key=lambda x: x['performance'])
        
        # Analyze top performers
        n_top = max(1, int(len(sorted_history) * top_percentile))
        top_performers = sorted_history[:n_top]
        
        # Extract patterns
        patterns = {
            'operation_frequencies': defaultdict(int),
            'combiner_frequencies': defaultdict(int),
            'avg_complexity': 0,
            'feature_usage': defaultdict(int),
            'successful_sequences': []
        }
        
        for arch in top_performers:
            nodes = arch['nodes']
            patterns['avg_complexity'] += len(nodes)
            
            for node in nodes:
                op = node.get('operation', '')
                combiner = node.get('combiner', '')
                
                patterns['operation_frequencies'][op] += 1
                patterns['combiner_frequencies'][combiner] += 1
                
                # Feature usage
                if 'SelectFeatures' in op:
                    hp = node.get('hyperparameters', {})
                    feat = tuple(hp.get('feature_indices', []))
                    patterns['feature_usage'][feat] += 1
        
        patterns['avg_complexity'] /= len(top_performers)
        
        # Convert to regular dicts for JSON serialization
        patterns['operation_frequencies'] = dict(patterns['operation_frequencies'])
        patterns['combiner_frequencies'] = dict(patterns['combiner_frequencies'])
        patterns['feature_usage'] = {str(k): v for k, v in patterns['feature_usage'].items()}
        
        return patterns


def create_generic_analysis_system(base_policy):
    """
    Enhance policy with generic mathematical analysis.
    
    NO hardcoded formulas - discovers patterns from data.
    
    Parameters
    ----------
    base_policy : MultiAgentDAGPolicy
        Base policy to enhance
    
    Returns
    -------
    MultiAgentDAGPolicy
        Enhanced policy with:
        - Redundancy detection
        - Formula deduplication
        - Pattern discovery
    """
    
    # Create analyzer
    analyzer = GenericFormulaAnalyzer()
    deduplicator = FormulaDeduplicator(analyzer)
    pattern_engine = PatternDiscoveryEngine()
    
    # Attach to policy
    base_policy.formula_analyzer = analyzer
    base_policy.formula_deduplicator = deduplicator
    base_policy.pattern_discovery = pattern_engine
    
    print("✅ Generic mathematical analysis activated")
    print("   Features:")
    print("     • Redundancy detection (x-x, x/x, x*1, etc.)")
    print("     • Formula deduplication (avoid re-evaluation)")
    print("     • Pattern discovery (learn from top performers)")
    print("     • NO hardcoded formulas - fully generic")
    
    return base_policy
