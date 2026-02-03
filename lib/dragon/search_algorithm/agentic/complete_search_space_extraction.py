"""
Complete Operation Extraction for Dragon Search Spaces
======================================================

Extracts ALL search space components:
- Operations (Identity, MLP, Conv, Attention, symbolic ops, etc.)
- Combiners (add, concat, mul)
- Activations (ReLU, GELU, Tanh, Identity, etc.)
- Hyperparameters and their possible values
"""

from typing import List, Dict, Set, Any, Tuple
from dragon.search_space.dag_variables import EvoDagVariable, HpVar, NodeVariable
from dragon.search_space.base_variables import CatVar, Constant, ArrayVar, DynamicBlock
from dragon.utils.tools import logger
import inspect
import torch.nn as nn


class CompleteSearchSpaceExtractor:
    """
    Complete extractor for all searchable components in Dragon search spaces.
    
    Extracts:
    - Operations: The actual layers/operations (MLP, Conv, Attention, symbolic ops)
    - Combiners: How inputs are combined (add, concat, mul)
    - Activations: Activation functions (ReLU, GELU, Tanh, etc.)
    - Hyperparameters: Configurable parameters for each operation
    """
    
    def __init__(self, search_space):
        self.search_space = search_space
        
        # Extracted components
        self.operations: Set[str] = set()
        self.combiners: Set[str] = set()
        self.activations: Set[str] = set()
        
        # Detailed information
        self.operation_details: Dict[str, Dict] = {}
        self.hyperparameter_spaces: Dict[str, Dict] = {}
    
    def extract_all(self) -> Dict[str, Any]:
        """
        Extract all searchable components.
        
        Returns
        -------
        dict
            {
                'operations': list[str],
                'combiners': list[str],
                'activations': list[str],
                'operation_details': dict,
                'hyperparameter_spaces': dict
            }
        """
        
        logger.info("Extracting complete search space...")
        
        # Multiple extraction strategies
        self._extract_from_structure()
        self._extract_from_samples()
        self._extract_combiners()
        self._extract_activations()
        
        # Clean and format
        result = {
            'operations': sorted(list(self.operations)),
            'combiners': sorted(list(self.combiners)),
            'activations': sorted(list(self.activations)),
            'operation_details': self.operation_details,
            'hyperparameter_spaces': self.hyperparameter_spaces
        }
        
        # Log summary
        logger.info(f"Extracted {len(result['operations'])} operations: {result['operations']}")
        logger.info(f"Extracted {len(result['combiners'])} combiners: {result['combiners']}")
        logger.info(f"Extracted {len(result['activations'])} activations: {result['activations']}")
        
        return result
    
    # ========================================================================
    # OPERATIONS EXTRACTION
    # ========================================================================
    
    def _extract_from_structure(self):
        """Extract operations from search space structure."""
        if isinstance(self.search_space, ArrayVar):
            for var in self.search_space.values:
                self._process_variable(var)
    
    def _process_variable(self, var):
        """Process a variable recursively."""
        
        if isinstance(var, EvoDagVariable):
            self._extract_from_dag_variable(var)
        elif isinstance(var, NodeVariable):
            self._extract_from_node_variable(var)
        elif isinstance(var, HpVar):
            self._extract_from_hp_var(var)
        elif isinstance(var, CatVar):
            self._extract_from_cat_var(var)
        elif isinstance(var, Constant):
            self._extract_from_constant(var)
    
    def _extract_from_dag_variable(self, dag_var: EvoDagVariable):
        """Extract from EvoDagVariable."""
        if hasattr(dag_var, 'operations') and hasattr(dag_var.operations, 'value'):
            self._process_variable(dag_var.operations.value)
    
    def _extract_from_node_variable(self, node_var: NodeVariable):
        """Extract from NodeVariable."""
        if hasattr(node_var, 'operation'):
            self._process_variable(node_var.operation)
        
        # Extract combiner options
        if hasattr(node_var, 'combiner'):
            self._extract_combiner_from_var(node_var.combiner)
        
        # Extract activation options
        if hasattr(node_var, 'activation_function'):
            self._extract_activation_from_var(node_var.activation_function)
    
    def _extract_from_hp_var(self, hp_var: HpVar):
        """Extract from HpVar."""
        
        # Extract operation
        if hasattr(hp_var, 'operation'):
            self._process_variable(hp_var.operation)
        
        # Extract hyperparameters
        if hasattr(hp_var, 'hyperparameters'):
            for hp_name, hp_def in hp_var.hyperparameters.items():
                self._extract_hyperparameter(hp_name, hp_def)
    
    def _extract_from_cat_var(self, cat_var: CatVar):
        """Extract from CatVar."""
        if not hasattr(cat_var, 'features'):
            return
        
        for feature in cat_var.features:
            if inspect.isclass(feature):
                # Direct operation class
                op_name = self._clean_operation_name(feature)
                self.operations.add(op_name)
                self.operation_details[op_name] = {
                    'class': feature,
                    'type': 'direct_class'
                }
            elif isinstance(feature, (HpVar, CatVar)):
                self._process_variable(feature)
    
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
    
    # ========================================================================
    # COMBINERS EXTRACTION
    # ========================================================================
    
    def _extract_combiners(self):
        """Extract combiner options from search space."""
        
        # Strategy 1: From NodeVariable definitions
        if isinstance(self.search_space, ArrayVar):
            for var in self.search_space.values:
                self._extract_combiner_recursive(var)
        
        # Strategy 2: From samples
        try:
            sample = self.search_space.random()
            if isinstance(sample, list):
                for item in sample:
                    self._extract_combiner_from_sample(item)
            else:
                self._extract_combiner_from_sample(sample)
        except Exception as e:
            logger.debug(f"Could not extract combiners from sample: {e}")
        
        # Default combiners if none found
        if not self.combiners:
            self.combiners = {'add', 'concat', 'mul'}
            logger.info("Using default combiners: add, concat, mul")
    
    def _extract_combiner_recursive(self, var):
        """Recursively extract combiners."""
        
        # NodeVariable has combiner
        if isinstance(var, NodeVariable) and hasattr(var, 'combiner'):
            self._extract_combiner_from_var(var.combiner)
        
        # EvoDagVariable
        if isinstance(var, EvoDagVariable) and hasattr(var, 'operations'):
            if hasattr(var.operations, 'value'):
                self._extract_combiner_recursive(var.operations.value)
        
        # HpVar, CatVar - check nested
        if hasattr(var, 'operation'):
            self._extract_combiner_recursive(var.operation)
        if hasattr(var, 'features'):
            for feat in var.features:
                if hasattr(feat, '__dict__'):
                    self._extract_combiner_recursive(feat)
    
    def _extract_combiner_from_var(self, combiner_var):
        """Extract combiners from a combiner variable."""
        
        # CatVar with combiner features
        if isinstance(combiner_var, CatVar) and hasattr(combiner_var, 'features'):
            for feature in combiner_var.features:
                if isinstance(feature, str):
                    self.combiners.add(feature)
        
        # Constant combiner
        elif isinstance(combiner_var, Constant) and hasattr(combiner_var, 'value'):
            if isinstance(combiner_var.value, str):
                self.combiners.add(combiner_var.value)
        
        # Direct string
        elif isinstance(combiner_var, str):
            self.combiners.add(combiner_var)
    
    def _extract_combiner_from_sample(self, sample):
        """Extract combiners from a sampled configuration."""
        
        if hasattr(sample, 'operations'):
            for node in sample.operations:
                if hasattr(node, 'combiner'):
                    combiner = node.combiner
                    if isinstance(combiner, str):
                        self.combiners.add(combiner)
    
    # ========================================================================
    # ACTIVATIONS EXTRACTION
    # ========================================================================
    
    def _extract_activations(self):
        """Extract activation function options."""
        
        # Strategy 1: From NodeVariable definitions
        if isinstance(self.search_space, ArrayVar):
            for var in self.search_space.values:
                self._extract_activation_recursive(var)
        
        # Strategy 2: From samples
        try:
            sample = self.search_space.random()
            if isinstance(sample, list):
                for item in sample:
                    self._extract_activation_from_sample(item)
            else:
                self._extract_activation_from_sample(sample)
        except Exception as e:
            logger.debug(f"Could not extract activations from sample: {e}")
        
        # Default activations if none found
        if not self.activations:
            self.activations = {'Identity', 'ReLU', 'GELU', 'Tanh', 'Sigmoid'}
            logger.info("Using default activations")
    
    def _extract_activation_recursive(self, var):
        """Recursively extract activations."""
        
        # NodeVariable has activation_function
        if isinstance(var, NodeVariable) and hasattr(var, 'activation_function'):
            self._extract_activation_from_var(var.activation_function)
        
        # Recurse through structure
        if isinstance(var, EvoDagVariable) and hasattr(var, 'operations'):
            if hasattr(var.operations, 'value'):
                self._extract_activation_recursive(var.operations.value)
    
    def _extract_activation_from_var(self, activation_var):
        """Extract activations from activation variable."""
        
        # CatVar with activation features
        if isinstance(activation_var, CatVar) and hasattr(activation_var, 'features'):
            for feature in activation_var.features:
                act_name = self._clean_activation_name(feature)
                if act_name:
                    self.activations.add(act_name)
        
        # Constant activation
        elif isinstance(activation_var, Constant) and hasattr(activation_var, 'value'):
            act_name = self._clean_activation_name(activation_var.value)
            if act_name:
                self.activations.add(act_name)
        
        # Direct activation object
        else:
            act_name = self._clean_activation_name(activation_var)
            if act_name:
                self.activations.add(act_name)
    
    def _extract_activation_from_sample(self, sample):
        """Extract activations from sampled configuration."""
        
        if hasattr(sample, 'operations'):
            for node in sample.operations:
                if hasattr(node, 'activation'):
                    act_name = self._clean_activation_name(node.activation)
                    if act_name:
                        self.activations.add(act_name)
    
    # ========================================================================
    # HYPERPARAMETERS EXTRACTION
    # ========================================================================
    
    def _extract_hyperparameter(self, hp_name: str, hp_def):
        """Extract hyperparameter definition."""
        
        hp_info = {
            'name': hp_name,
            'type': type(hp_def).__name__
        }
        
        # CatVar - discrete choices
        if isinstance(hp_def, CatVar) and hasattr(hp_def, 'features'):
            hp_info['type'] = 'categorical'
            hp_info['choices'] = hp_def.features
        
        # IntVar - integer range
        elif hasattr(hp_def, 'b_min') and hasattr(hp_def, 'b_max'):
            hp_info['type'] = 'integer'
            hp_info['min'] = hp_def.b_min
            hp_info['max'] = hp_def.b_max
        
        # FloatVar - float range
        elif hasattr(hp_def, 'min') and hasattr(hp_def, 'max'):
            hp_info['type'] = 'float'
            hp_info['min'] = hp_def.min
            hp_info['max'] = hp_def.max
        
        self.hyperparameter_spaces[hp_name] = hp_info
    
    # ========================================================================
    # SAMPLING-BASED EXTRACTION
    # ========================================================================
    
    def _extract_from_samples(self, n_samples=3):
        """Extract by sampling the search space multiple times."""
        
        for i in range(n_samples):
            try:
                sample = self.search_space.random()
                
                if isinstance(sample, list):
                    for item in sample:
                        self._extract_from_sample_item(item)
                else:
                    self._extract_from_sample_item(sample)
            
            except Exception as e:
                logger.debug(f"Sample {i} extraction failed: {e}")
    
    def _extract_from_sample_item(self, item):
        """Extract all info from a sampled item."""
        
        if hasattr(item, 'operations'):
            for node in item.operations:
                # Operation
                if hasattr(node, 'name'):
                    op_name = self._clean_operation_name(node.name)
                    self.operations.add(op_name)
                    
                    # Hyperparameters
                    if hasattr(node, 'hp') and node.hp:
                        if op_name not in self.operation_details:
                            self.operation_details[op_name] = {}
                        self.operation_details[op_name]['hyperparameters'] = list(node.hp.keys())
                
                # Combiner
                if hasattr(node, 'combiner'):
                    if isinstance(node.combiner, str):
                        self.combiners.add(node.combiner)
                
                # Activation
                if hasattr(node, 'activation'):
                    act_name = self._clean_activation_name(node.activation)
                    if act_name:
                        self.activations.add(act_name)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _clean_operation_name(self, op) -> str:
        """Clean operation name from class or string."""
        
        if inspect.isclass(op):
            return op.__name__
        
        op_str = str(op)
        
        if '<class' in op_str:
            parts = op_str.split('.')
            if parts:
                return parts[-1].replace("'>", "").replace('">', '')
        
        if '.' in op_str:
            return op_str.split('.')[-1]
        
        return op_str
    
    def _clean_activation_name(self, activation) -> str:
        """
        Clean activation name from PyTorch module or string.
        
        Examples:
        - ReLU() → ReLU
        - GELU(approximate='none') → GELU
        - Identity() → Identity
        """
        
        if activation is None:
            return ""
        
        # PyTorch module
        if hasattr(activation, '__class__'):
            class_name = activation.__class__.__name__
            # Filter out non-activation classes
            if class_name in ['str', 'int', 'float', 'list', 'dict']:
                return ""
            return class_name
        
        # String
        act_str = str(activation)
        
        # Remove parentheses and parameters
        if '(' in act_str:
            act_str = act_str.split('(')[0]
        
        # Clean class path
        if '.' in act_str:
            act_str = act_str.split('.')[-1]
        
        return act_str


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def extract_complete_search_space(search_space) -> Dict[str, Any]:
    """
    Extract all searchable components from Dragon search space.
    
    Parameters
    ----------
    search_space : ArrayVar
        Dragon search space
    
    Returns
    -------
    dict
        {
            'operations': ['Identity', 'MLP', 'Conv1D', ...],
            'combiners': ['add', 'concat', 'mul'],
            'activations': ['ReLU', 'GELU', 'Tanh', ...],
            'operation_details': {...},
            'hyperparameter_spaces': {...}
        }
    """
    
    extractor = CompleteSearchSpaceExtractor(search_space)
    return extractor.extract_all()


def get_complete_summary(search_space) -> str:
    """
    Get formatted summary of complete search space.
    
    Parameters
    ----------
    search_space : ArrayVar
        Search space
    
    Returns
    -------
    str
        Formatted summary
    """
    
    info = extract_complete_search_space(search_space)
    
    summary = "="*70 + "\n"
    summary += " "*20 + "SEARCH SPACE SUMMARY\n"
    summary += "="*70 + "\n\n"
    
    # Operations
    summary += f"OPERATIONS ({len(info['operations'])} total):\n"
    summary += "-"*70 + "\n"
    for op in info['operations']:
        hp_info = ""
        if op in info['operation_details']:
            if 'hyperparameters' in info['operation_details'][op]:
                hps = info['operation_details'][op]['hyperparameters']
                if hps:
                    hp_info = f" → HPs: {', '.join(hps)}"
        summary += f"  • {op}{hp_info}\n"
    
    # Combiners
    summary += f"\nCOMBINERS ({len(info['combiners'])} total):\n"
    summary += "-"*70 + "\n"
    for combiner in info['combiners']:
        summary += f"  • {combiner}\n"
    
    # Activations
    summary += f"\nACTIVATIONS ({len(info['activations'])} total):\n"
    summary += "-"*70 + "\n"
    for activation in info['activations']:
        summary += f"  • {activation}\n"
    
    # Hyperparameters
    if info['hyperparameter_spaces']:
        summary += f"\nHYPERPARAMETERS ({len(info['hyperparameter_spaces'])} total):\n"
        summary += "-"*70 + "\n"
        for hp_name, hp_info in info['hyperparameter_spaces'].items():
            summary += f"  • {hp_name}: {hp_info.get('type', 'unknown')}"
            if 'choices' in hp_info:
                summary += f" choices={hp_info['choices']}"
            elif 'min' in hp_info and 'max' in hp_info:
                summary += f" range=[{hp_info['min']}, {hp_info['max']}]"
            summary += "\n"
    
    summary += "\n" + "="*70 + "\n"
    
    return summary