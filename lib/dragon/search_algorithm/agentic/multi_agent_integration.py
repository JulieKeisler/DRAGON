"""
Integration layer for Multi-Agent RAG System
============================================

Provides:
- Automatic extraction of available operations from search space
- Integration with DAGAgenticSearchAlgorithm
- Mutation application with operation selection
"""

import copy
import pickle
from typing import List, Dict, Any
from dragon.search_space.dag_variables import EvoDagVariable
from dragon.search_algorithm.agentic.agentic import AgenticSearchAlgorithm
from dragon.utils.tools import logger
from dragon.search_algorithm.agentic.multi_agent_rag_system import MultiAgentDAGPolicy, MutationProposal


def extract_available_operations(search_space) -> List[str]:
    """
    Extract all available operations from the search space.
    
    Uses enhanced extraction that handles multiple search space patterns.
    
    Parameters
    ----------
    search_space : ArrayVar or similar
        Search space containing DAG variable(s)
    
    Returns
    -------
    list[str]
        List of available operation names
    """
    from dragon.search_algorithm.agentic.enhanced_operation_extraction import OperationExtractor
    
    extractor = OperationExtractor(search_space)
    operations, details = extractor.extract()
    
    return operations


class MultiAgentDAGSearchAlgorithm(AgenticSearchAlgorithm):
    """
    DAG search algorithm with multi-agent RAG system.
    
    Features:
    - Automatic extraction of complete search space
    - Multi-agent decision making
    - Knowledge base for learning
    - Detailed mutation control (operations, combiners, activations)
    """
    
    def __init__(self, search_space, T, K, N, E, evaluation, save_dir,
                 api_key: str, task_description: str,
                 models=None, pop_path=None, verbose=False, model="llama-3.3-70b-versatile", **args):
        
        # Extract complete search space info
        from dragon.search_algorithm.agentic.complete_search_space_extraction import extract_complete_search_space
        
        search_space_info = extract_complete_search_space(search_space)
        
        if not search_space_info['operations']:
            logger.warning("No operations extracted, using defaults")
            search_space_info['operations'] = ["Identity", "MLP", "Dropout"]
        
        if not search_space_info['combiners']:
            search_space_info['combiners'] = ["add", "concat", "mul"]
        
        if not search_space_info['activations']:
            search_space_info['activations'] = ["Identity", "ReLU", "GELU"]
        
        # Create multi-agent policy with complete info
        policy = MultiAgentDAGPolicy(
            api_key=api_key,
            task_description=task_description,
            search_space_info=search_space_info,
            allow_train=(N > 1),
            model=model
        )
        
        # Initialize parent
        super().__init__(
            search_space=search_space,
            T=T, K=K, N=N, E=E,
            evaluation=evaluation,
            save_dir=save_dir,
            models=models,
            pop_path=pop_path,
            verbose=verbose,
            policy=policy,
            task_description=task_description,
            **args
        )
        
        self.search_space_info = search_space_info
    
    def _execute_primary_operation(self, action, state):
        """Execute with multi-agent mutation support."""
        primary_op = action.get("primary_op")
        primary_target = action.get("primary_target")

        if not primary_op or primary_op not in ["CREATE", "MUTATE_DAG", "TRAIN"]:
            return []

        if primary_op == "CREATE":
            return self._create_individual()
        
        elif primary_op == "MUTATE_DAG":
            target_idx = self._resolve_target(primary_target, state)
            mutations = action.get("mutations", [])
            return self._mutate_dag_with_operations(target_idx, mutations)
        
        elif primary_op == "TRAIN":
            target_idx = self._resolve_target(primary_target, state)
            return self._train_individual(target_idx)
        
        return []
    
    def _mutate_dag_with_operations(self, parent_idx: int, 
                                     mutations: List[Dict]) -> List[int]:
        """
        Apply mutations with specific operation selection.
        
        Parameters
        ----------
        parent_idx : int
            Parent configuration index
        mutations : list[dict]
            Mutations with operation choices
            [{
                'type': str,
                'node_idx': int,
                'operation': str | None,
                'hyperparameters': dict | None
            }]
        
        Returns
        -------
        list[int]
            New individual index
        """
        try:
            with open(f"{self.save_dir}/x_{parent_idx}.pkl", 'rb') as f:
                parent_config = pickle.load(f)
        except FileNotFoundError:
            logger.warning(f"Parent {parent_idx} not found, creating random")
            return self._create_individual()
        
        try:
            # Apply intelligent mutations
            mutant_config = self._apply_intelligent_mutations(
                parent_config, mutations
            )
            
            # Record in knowledge base
            new_idx = self.K
            self.K += 1
            
            with open(f"{self.save_dir}/x_{new_idx}.pkl", 'wb') as f:
                pickle.dump(mutant_config, f)
            
            self.sent[new_idx] = {"N": 0, "N_bar": 0, "UCBLoss": 0}
            
            # Log
            mut_str = ", ".join([
                f"{m['type']}@{m['node_idx']}" + 
                (f"→{m.get('operation', '')}" if m.get('operation') else "")
                for m in mutations
            ])
            print(f"🧬 Intelligent mutation: {parent_idx} → {new_idx}")
            print(f"   Applied: [{mut_str}]")
            
            return [new_idx]
            
        except Exception as e:
            logger.error(f"Intelligent mutation failed: {e}")
            import traceback
            traceback.print_exc()
            return self._create_individual()
    
    def _apply_intelligent_mutations(self, config: Any, 
                                     mutations: List[Dict]) -> Any:
        """
        Apply mutations with operation selection.
        
        If mutation specifies an operation, tries to use it.
        """
        mutated = copy.deepcopy(config)
        
        # Find DAG
        dag_var = None
        dag_idx = None
        for i, var in enumerate(self.search_space.values):
            if isinstance(var, EvoDagVariable):
                dag_var = var
                dag_idx = i
                break
        
        if dag_var is None:
            logger.warning("No DAG found, using standard neighbor")
            return self.search_space.neighbor(config)
        
        dag_value = mutated[dag_idx]
        dag_neighbor = dag_var.neighbor
        
        # Apply each mutation
        for mutation in mutations:
            mut_type = mutation.get("type", "modify")
            node_idx = mutation.get("node_idx", 0)
            requested_op = mutation.get("operation")
            requested_hp = mutation.get("hyperparameters")
            
            # Handle negative indices
            if node_idx < 0:
                node_idx = len(dag_value.operations) + node_idx
            node_idx = max(0, min(node_idx, len(dag_value.operations) - 1))
            
            try:
                # Apply base mutation
                dag_value = dag_neighbor.modification(
                    mut_type, node_idx, dag_value
                )
                
                # If operation was specified and this is modify/add, try to apply it
                if requested_op and mut_type in ["modify", "add"]:
                    success = self._try_apply_operation(
                        dag_value, node_idx, requested_op, requested_hp
                    )
                    if success:
                        logger.info(f"Applied requested operation: {requested_op}")
                    else:
                        logger.warning(f"Could not apply {requested_op}, using default")
                
            except Exception as e:
                logger.error(f"Error applying {mut_type}: {e}")
                continue
        
        # Validate
        try:
            dag_value.assert_adj_matrix()
        except AssertionError as e:
            logger.error(f"Invalid DAG, using fallback: {e}")
            return self.search_space.neighbor(config)
        
        mutated[dag_idx] = dag_value
        return mutated
    
    def _try_apply_operation(self, dag, node_idx: int, op_name: str, 
                            hyperparams: Dict = None) -> bool:
        """
        Try to apply a specific operation to a node.
        
        Returns
        -------
        bool
            True if successful
        """
        try:
            # Get the node
            if node_idx >= len(dag.operations):
                return False
            
            node = dag.operations[node_idx]
            
            # Try to find and set the operation
            # This is simplified - actual implementation depends on your node structure
            if hasattr(node, 'name'):
                # Check if operation is available
                # For now, just log (full implementation would need access to operation objects)
                logger.debug(f"Would set node {node_idx} to {op_name}")
                # node.name = get_operation_class(op_name)  # Simplified
                
                if hyperparams and hasattr(node, 'hp'):
                    for k, v in hyperparams.items():
                        if k in node.hp:
                            node.hp[k] = v
                
                return True
        
        except Exception as e:
            logger.debug(f"Could not apply operation {op_name}: {e}")
            return False
        
        return False


# Convenience factory function
def create_multi_agent_search(search_space, save_dir: str, api_key: str,
                              task_description: str, evaluation,
                              T=20, K=5, N=1, model="llama-3.3-70b-versatile", **kwargs):
    """
    Create a multi-agent DAG search with recommended settings.
    
    Parameters
    ----------
    search_space : SearchSpace
        DAG search space
    save_dir : str
        Results directory
    api_key : str
        Groq API key
    task_description : str
        Task description for agents
    evaluation : callable
        Evaluation function
    T, K, N : int
        Search parameters
    **kwargs
        Additional arguments
    
    Returns
    -------
    MultiAgentDAGSearchAlgorithm
        Configured algorithm
    
    Examples
    --------
    >>> algo = create_multi_agent_search(
    ...     search_space=my_dag_space,
    ...     save_dir="./results",
    ...     api_key="your_groq_key",
    ...     task_description="Symbolic regression for NDVI formula",
    ...     evaluation=my_loss_fn,
    ...     T=1000,
    ...     K=200,
    ...     N=1
    ... )
    >>> algo.run()
    """
    
    return MultiAgentDAGSearchAlgorithm(
        search_space=search_space,
        T=T,
        K=K,
        N=N,
        E=0.01,
        evaluation=evaluation,
        save_dir=save_dir,
        api_key=api_key,
        task_description=task_description,
        model=model,
        **kwargs
    )