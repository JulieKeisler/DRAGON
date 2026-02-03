"""
Enhanced Operation Extraction for Dragon Search Spaces
======================================================

Handles multiple search space patterns:
- ArrayVar with DAG variables
- Complex nested structures with HpVar, CatVar, Constant
- Multiple DAG blocks (2D, 1D, etc.)
- Custom operation definitions
"""

from typing import List, Dict, Set, Any, Tuple
from dragon.search_space.dag_variables import EvoDagVariable, HpVar, NodeVariable
from dragon.search_space.base_variables import CatVar, Constant, ArrayVar, DynamicBlock
from dragon.utils.tools import logger
import inspect


class OperationExtractor:
    """
    Intelligent extractor for available operations from Dragon search spaces.
    
    Handles multiple patterns and nested structures.
    Extracts: operations, combiners, activations, and hyperparameters.
    """
    
    def __init__(self, search_space):
        self.search_space = search_space
        self.operations: Set[str] = set()
        self.combiners: Set[str] = set()
        self.activations: Set[str] = set()
        self.operation_details: Dict[str, Dict] = {}
        self.search_space_info: Dict[str, Any] = {}
        
    def extract(self) -> Tuple[List[str], Dict[str, Dict]]:
        """
        Extract all available operations.
        
        Returns
        -------
        tuple
            (operation_names, operation_details)
            - operation_names: List of operation names
            - operation_details: Dict with hyperparameter info
        """
        
        # Multiple extraction strategies
        self._extract_from_array_var()
        self._extract_from_sample()
        self._extract_from_variable_structure()
        
        # Clean and sort
        operations = sorted(list(self.operations))
        
        if operations:
            logger.info(f"Extracted {len(operations)} operations: {operations}")
        else:
            logger.warning("No operations extracted, using defaults")
            operations = ["Identity", "MLP", "Dropout"]
        
        return operations, self.operation_details
    
    def _extract_from_array_var(self):
        """Extract from ArrayVar structure."""
        if not isinstance(self.search_space, ArrayVar):
            return
        
        # Iterate through values in ArrayVar
        for var in self.search_space.values:
            self._process_variable(var)
    
    def _process_variable(self, var):
        """Process a variable recursively."""
        
        # EvoDagVariable
        if isinstance(var, EvoDagVariable):
            self._extract_from_dag_variable(var)
        
        # NodeVariable
        elif isinstance(var, NodeVariable):
            self._extract_from_node_variable(var)
        
        # HpVar
        elif isinstance(var, HpVar):
            self._extract_from_hp_var(var)
        
        # CatVar
        elif isinstance(var, CatVar):
            self._extract_from_cat_var(var)
        
        # Constant
        elif isinstance(var, Constant):
            self._extract_from_constant(var)
    
    def _extract_from_dag_variable(self, dag_var: EvoDagVariable):
        """Extract from EvoDagVariable."""
        
        # Access operations through DynamicBlock
        if hasattr(dag_var, 'operations'):
            ops_var = dag_var.operations
            
            # DynamicBlock contains the value (NodeVariable)
            if hasattr(ops_var, 'value'):
                self._process_variable(ops_var.value)
    
    def _extract_from_node_variable(self, node_var: NodeVariable):
        """Extract from NodeVariable (contains operations)."""
        
        # NodeVariable has 'operation' attribute
        if hasattr(node_var, 'operation'):
            self._process_variable(node_var.operation)
    
    def _extract_from_hp_var(self, hp_var: HpVar):
        """Extract from HpVar."""
        
        # HpVar has 'operation' which can be CatVar or Constant
        if hasattr(hp_var, 'operation'):
            op_var = hp_var.operation
            self._process_variable(op_var)
        
        # Also check hyperparameters for nested structures
        if hasattr(hp_var, 'hyperparameters'):
            for hp_name, hp_var_nested in hp_var.hyperparameters.items():
                if isinstance(hp_var_nested, CatVar):
                    # Hyperparameter might contain operation choices
                    pass
    
    def _extract_from_cat_var(self, cat_var: CatVar):
        """Extract from CatVar (list of features)."""
        
        if not hasattr(cat_var, 'features'):
            return
        
        for feature in cat_var.features:
            # Feature can be:
            # - A class (operation class)
            # - An HpVar
            # - Another CatVar
            # - A list (for hyperparameter choices)
            
            if inspect.isclass(feature):
                # It's an operation class
                op_name = self._clean_operation_name(feature)
                self.operations.add(op_name)
                self.operation_details[op_name] = {
                    'class': feature,
                    'type': 'direct_class'
                }
            
            elif isinstance(feature, HpVar):
                self._extract_from_hp_var(feature)
            
            elif isinstance(feature, CatVar):
                self._extract_from_cat_var(feature)
            
            elif isinstance(feature, list):
                # List of choices (like [[0], [1]] for feature indices)
                pass
    
    def _extract_from_constant(self, const_var: Constant):
        """Extract from Constant."""
        
        if hasattr(const_var, 'value'):
            value = const_var.value
            
            if inspect.isclass(value):
                op_name = self._clean_operation_name(value)
                self.operations.add(op_name)
                self.operation_details[op_name] = {
                    'class': value,
                    'type': 'constant'
                }
    
    def _extract_from_sample(self):
        """Extract by sampling the search space."""
        
        try:
            # Try to generate a sample
            sample = self.search_space.random()
            
            # If sample is a list (ArrayVar result)
            if isinstance(sample, list):
                for item in sample:
                    self._extract_from_sample_item(item)
            else:
                self._extract_from_sample_item(sample)
        
        except Exception as e:
            logger.debug(f"Could not extract from sample: {e}")
    
    def _extract_from_sample_item(self, item):
        """Extract operations from a sampled item."""
        
        # Check if it's a DAG with operations
        if hasattr(item, 'operations'):
            for node in item.operations:
                if hasattr(node, 'name'):
                    op_name = self._clean_operation_name(node.name)
                    self.operations.add(op_name)
                    
                    # Extract hyperparameters info
                    if hasattr(node, 'hp') and node.hp:
                        if op_name not in self.operation_details:
                            self.operation_details[op_name] = {}
                        self.operation_details[op_name]['hyperparameters'] = list(node.hp.keys())
    
    def _extract_from_variable_structure(self):
        """
        Extract by analyzing variable structure directly.
        
        Useful for symbolic regression and custom operations.
        """
        
        if not isinstance(self.search_space, ArrayVar):
            return
        
        # Deep inspection of structure
        for var in self.search_space.values:
            self._deep_inspect(var)
    
    def _deep_inspect(self, obj, depth=0, max_depth=5):
        """Deep recursive inspection of object structure."""
        
        if depth > max_depth:
            return
        
        # Check common attributes
        for attr_name in ['operation', 'operations', 'features', 'value', 'values', 'candidates']:
            if hasattr(obj, attr_name):
                attr_value = getattr(obj, attr_name)
                
                # Process based on type
                if inspect.isclass(attr_value):
                    op_name = self._clean_operation_name(attr_value)
                    self.operations.add(op_name)
                
                elif isinstance(attr_value, (list, tuple)):
                    for item in attr_value:
                        if inspect.isclass(item):
                            op_name = self._clean_operation_name(item)
                            self.operations.add(op_name)
                        else:
                            self._deep_inspect(item, depth + 1, max_depth)
                
                elif hasattr(attr_value, '__dict__'):
                    self._deep_inspect(attr_value, depth + 1, max_depth)
    
    def _clean_operation_name(self, op) -> str:
        """
        Clean operation name from class or string.
        
        Examples:
        - <class 'dragon.search_space.bricks.basics.MLP'> → MLP
        - SelectFeatures → SelectFeatures
        - dragon.search_space.bricks.symbolic_regression.Inverse → Inverse
        """
        
        if inspect.isclass(op):
            return op.__name__
        
        op_str = str(op)
        
        # Handle <class 'package.module.ClassName'>
        if '<class' in op_str:
            # Extract last part after last dot
            parts = op_str.split('.')
            if parts:
                name = parts[-1].replace("'>", "").replace('">', '')
                return name
        
        # Handle module.ClassName format
        if '.' in op_str:
            return op_str.split('.')[-1]
        
        return op_str


def extract_available_operations(search_space) -> Tuple[List[str], Dict[str, Dict]]:
    """
    Extract all available operations from a Dragon search space.
    
    Works with multiple patterns:
    - Simple DAG with operations_var
    - Complex nested structures
    - Symbolic regression operations
    - Multiple DAG blocks (2D, 1D)
    
    Parameters
    ----------
    search_space : ArrayVar or SearchSpace
        Dragon search space
    
    Returns
    -------
    tuple
        (operation_names, operation_details)
        
    Examples
    --------
    >>> # Pattern 1: Symbolic regression
    >>> ops, details = extract_available_operations(search_space)
    >>> ops
    ['Identity', 'Inverse', 'Negate', 'SelectFeatures', 'SumFeatures']
    
    >>> # Pattern 2: Deep learning
    >>> ops, details = extract_available_operations(search_space)
    >>> ops
    ['Identity', 'Attention1D', 'Attention2D', 'MLP', 'Conv1D', 'Conv2D', 
     'MaxPooling1D', 'AVGPooling1D', 'Dropout', 'LayerNorm', 'BatchNorm']
    """
    
    extractor = OperationExtractor(search_space)
    return extractor.extract()


def get_operation_summary(search_space) -> str:
    """
    Get a human-readable summary of available operations.
    
    Parameters
    ----------
    search_space : ArrayVar
        Search space
    
    Returns
    -------
    str
        Formatted summary
    """
    
    ops, details = extract_available_operations(search_space)
    
    summary = f"Available Operations ({len(ops)} total):\n"
    summary += "="*50 + "\n\n"
    
    # Group by type if possible
    groups = {
        'basic': [],
        'attention': [],
        'conv': [],
        'pooling': [],
        'norm': [],
        'symbolic': [],
        'other': []
    }
    
    for op in ops:
        op_lower = op.lower()
        if 'attention' in op_lower:
            groups['attention'].append(op)
        elif 'conv' in op_lower:
            groups['conv'].append(op)
        elif 'pool' in op_lower:
            groups['pooling'].append(op)
        elif 'norm' in op_lower or 'batch' in op_lower or 'layer' in op_lower:
            groups['norm'].append(op)
        elif any(x in op_lower for x in ['identity', 'mlp', 'dropout']):
            groups['basic'].append(op)
        elif any(x in op_lower for x in ['inverse', 'negate', 'sum', 'select']):
            groups['symbolic'].append(op)
        else:
            groups['other'].append(op)
    
    for group_name, group_ops in groups.items():
        if group_ops:
            summary += f"{group_name.upper()}:\n"
            for op in sorted(group_ops):
                hp_info = ""
                if op in details and 'hyperparameters' in details[op]:
                    hps = details[op]['hyperparameters']
                    if hps:
                        hp_info = f" (HPs: {', '.join(hps)})"
                summary += f"  - {op}{hp_info}\n"
            summary += "\n"
    
    return summary
