"""
Advanced Agentic Mutations
===========================
Extends the agentic search to give LLM control over DAG mutations.
Extension of AgenticSearchAlgorithm to support LLM-controlled DAG mutations
"""

import copy
from dragon.search_algorithm.agentic import EnhancedLLMPolicy, AdaptiveLLMPolicy
from dragon.search_space.dag_variables import EvoDagVariable
from dragon.utils.tools import logger
import json
import numpy as np
from dragon.search_algorithm.agentic import AgenticSearchAlgorithm
from dragon.utils.tools import logger
import pickle


class DAGMutationPolicy(EnhancedLLMPolicy):
    """
    LLM policy with fine-grained control over DAG mutations.
    
    Instead of just MUTATE, the LLM can specify:
    - Which node to modify
    - What type of modification (add/delete/modify/children/parents)
    - Which hyperparameters to change
    
    Parameters
    ----------
    model : str, default="llama-3.3-70b-versatile"
        Groq model name
    temperature : float, default=0.3
        Sampling temperature
    api_key : str, optional
        Groq API key
    history_length : int, default=10
        History size
    allow_train : bool, default=True
        Whether TRAIN is allowed
    task_description : str, optional
        Task description
    mutation_types : list, optional
        Allowed mutation types for DAG
    """
    
    def __init__(self, model="llama-3.3-70b-versatile", temperature=0.3, api_key=None, 
                 history_length=10, allow_train=True, task_description=None,
                 mutation_types=None):
        super().__init__(model=model, temperature=temperature, api_key=api_key,
                        history_length=history_length, allow_train=allow_train,
                        task_description=task_description)
        
        if mutation_types is None:
            mutation_types = ["add", "delete", "modify", "children", "parents"]
        self.mutation_types = mutation_types

    def decide(self, state, search_space=None, save_dir=None, task_description=None):
        """Make intelligent decision with DAG-specific mutation control."""
        
        task_desc = task_description or self.task_description or "optimization"
        ops_available = ["MUTATE_DAG", "CREATE"] + (["TRAIN"] if self.allow_train else [])

        # Get architecture info if available
        arch_info = self._extract_architecture_info(state, save_dir)
        enriched_pop = self._enrich_population(state, save_dir)
        trends = self._compute_trends()
        performance_analysis = self._analyze_past_performance()
        observations = self._get_critical_observations(state)
        
        # Create safe population display
        pop_display = []
        for ind in enriched_pop:
            try:
                loss_val = float(ind["loss"]) if not isinstance(ind["loss"], (int, float)) else ind["loss"]
            except (TypeError, ValueError, AttributeError):
                loss_val = 0.0
            
            pop_display.append({
                "idx": int(ind["idx"]),
                "loss": round(loss_val, 6),
                "N": int(ind["N"]),
                "config": str(ind.get("config_preview", "N/A"))[:50]
            })

        prompt = self._build_dag_prompt(
            task_desc, state, pop_display, arch_info, trends,
            performance_analysis, observations, ops_available
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                response_format={"type": "json_object"}
            )

            action = json.loads(response.choices[0].message.content)
            action = self._validate_and_fix_dag_action(action, ops_available, state)
            
            self._log_dag_decision(action)
            self._add_to_history(state, action)
            
            return action

        except Exception as e:
            logger.error(f"LLM error: {e}")
            import traceback
            traceback.print_exc()
            
            fallback = self._create_fallback_action(state)
            self._add_to_history(state, fallback)
            return fallback

    def _extract_architecture_info(self, state, save_dir):
        """Extract detailed architecture information from configurations."""
        if not save_dir:
            return None
        
        arch_summary = []
        
        for ind in state['full_population']:  # Analyze top 3
            try:
                with open(f"{save_dir}/x_{ind['idx']}.pkl", 'rb') as f:
                    config = pickle.load(f)
                
                # Try to extract DAG structure
                summary = self._parse_dag_structure(config, ind['idx'])
                arch_summary.append(summary)
                
            except Exception as e:
                logger.debug(f"Could not extract architecture for idx {ind['idx']}: {e}")
        
        return arch_summary if arch_summary else None

    def _parse_dag_structure(self, config, idx):
        """Parse DAG structure into human-readable summary."""
        try:
            config_str = str(config)
            
            # Count nodes in each block
            blocks = []
            lines = config_str.split('\n')
            
            current_block = []
            for line in lines:
                if 'ModuleList' in line and current_block:
                    blocks.append(current_block)
                    current_block = []
                elif '(combiner)' in line or '(op)' in line:
                    current_block.append(line.strip())
            
            if current_block:
                blocks.append(current_block)
            
            summary = {
                "idx": idx,
                "num_blocks": len(blocks),
                "total_nodes": sum(len(b) for b in blocks),
                "blocks": [
                    {
                        "nodes": len(block),
                        "operations": self._extract_operations(block)
                    }
                    for block in blocks  # First 2 blocks only
                ],
                "archi": config
            }
            
            return summary
            
        except Exception as e:
            return {"idx": idx, "error": str(e)}

    def _extract_operations(self, block_lines):
        """Extract operation types from block."""
        ops = []
        for line in block_lines:
            if '(op)' in line:
                # Extract operation name
                parts = line.split('(op)')
                if len(parts) > 1:
                    op_part = parts[1].strip().split('(')[0].strip()
                    ops.append(op_part)
        return ops[:5]  # Limit to 5

    def _build_dag_prompt(self, task_desc, state, pop_display, arch_info, 
                          trends, performance, observations, ops_available):
        """Build specialized prompt for DAG mutations."""
        
        arch_section = ""
        if arch_info:
            arch_section = f"\n🏗️ ARCHITECTURE ANALYSIS (top configs):\n"
            for arch in arch_info[:2]:
                if 'error' not in arch:
                    arch_section += f"idx={arch['idx']}: {arch['num_blocks']} blocks, {arch['total_nodes']} total nodes\n"
                    for i, block in enumerate(arch['blocks']):
                        arch_section += f"  Block {i}: {block['nodes']} nodes, ops={block['operations']}\n"
        
        mutation_guide = f"""
MUTATION TYPES (for MUTATE_DAG):
- add: Add new node after target
- delete: Remove target node
- modify: Change operation/hyperparameters of target node
- children: Modify connections to children
- parents: Modify connections from parents

NODE SELECTION STRATEGIES:
- Early nodes (0-2): Control input processing, safe to modify
- Middle nodes: Core architecture, modify carefully
- Last node: Output layer, critical - avoid deletion
- Bottleneck nodes: High connectivity, high impact
"""

        mess = f"""You are an expert AI controlling evolutionary neural architecture search for {task_desc}.

📊 STATUS (Step {state['step']}/{state['max_steps']}, {(state['step']/state['max_steps']*100):.0f}%):
Best: idx={state['best']['idx']}, loss={state['best']['loss']:.6f}, N={state['best']['N']}
Worst: idx={state['worst']['idx']}, loss={state['worst']['loss']:.6f}, N={state['worst']['N']}
Stats: μ={state['stats']['mean_loss']:.6f}, σ={state['stats']['std_loss']:.6f}
Population: {state['population_size']} individuals
{arch_section}

Population sample:
{json.dumps(pop_display, indent=2)}

History: {self._format_history()}
Trends: {trends}
Performance: {performance}
{f"⚠️ {observations}" if observations != "No critical issues." else ""}

{mutation_guide}

TASK:
1. DROP:
Select 1-3 individuals to remove.
Prioritize removing configurations that:
- Show exploding or highly unstable loss
- Share structural patterns correlated with failure
- Repeatedly underperform across evaluations
Avoid removing individuals that differ structurally from known failure patterns.
2. PRIMARY: Choose operation:
   - MUTATE_DAG: Apply targeted DAG mutations (specify details below)
   - CREATE: Generate completely random architecture
   {f"- TRAIN: Continue training existing config" if self.allow_train else ""}

If MUTATE_DAG selected, specify:
- mutation_target: Which config to mutate ("best" | "worst" | "random" | <int>)
- node_idx: integer in range [-N, N-1], where:
  - 0 refers to the first node
  - -1 refers to the last node
  - other valid indices refer to intermediate nodes


Strategy:
- Early (<30%):
  Explore the space aggressively (CREATE, bold structural mutations).
  Focus on discovering diverse topologies and operation patterns.

- Mid (30-70%):
  Refine and exploit promising architectures.
  Identify high-performing sub-structures (node patterns, connectivity motifs)
  and apply targeted MUTATE_DAG to preserve and extend them
  (local changes, selective add/delete, connectivity adjustments).

- Late (>70%):
  Fine-tune the best architectures.
  Preserve core structure and only apply low-risk mutations.
  Prefer small, localized changes that improve performance without altering
  the overall topology (mainly careful modify).


Return ONLY JSON:
{{
    "drop_target": ["worst"] | [<int>, <int>] | [<int>, <int>, <int>],
    "primary_op": "MUTATE_DAG" | "CREATE" | {"TRAIN" if self.allow_train else "null"},
    "primary_target": "best" | "worst" | "random" | <int> | null,
    "mutations": [
        ["type": "add" | "delete" | "modify" | "children" | "parents",
        "node_idx": int   // any valid node index (0 = first, -1 = last)]
    ] (1 to 3 items) | null,
    "confidence": <float 0-1>,
    "reasoning": "<Explain decision and confidence value>"
}}"""
        #print(mess)
        return mess

    def _validate_and_fix_dag_action(self, action, ops_available, state):
        """Validate and fix DAG-specific action."""
        
        # Ensure drop_target is a list
        if "drop_target" not in action or action["drop_target"] is None:
            action["drop_target"] = ["worst"]
        elif not isinstance(action["drop_target"], list):
            action["drop_target"] = [action["drop_target"]]
        
        # Validate primary_op
        primary_op = action.get("primary_op")
        if primary_op not in ops_available and primary_op is not None:
            logger.warning(f"Invalid primary_op '{primary_op}', defaulting to CREATE")
            action["primary_op"] = "CREATE"
            action["primary_target"] = None
        
        # Validate primary_target if needed
        if action.get("primary_op") and action["primary_op"] != "CREATE":
            pt = action.get("primary_target")
            if pt not in ["best", "worst", "random"] and not isinstance(pt, int):
                logger.warning(f"Invalid primary_target '{pt}', defaulting to 'best'")
                action["primary_target"] = "best"
        
        # Validate confidence
        try:
            action["confidence"] = float(action.get("confidence", 0.5))
            action["confidence"] = max(0.0, min(1.0, action["confidence"]))
        except (TypeError, ValueError):
            action["confidence"] = 0.5
        
        # Ensure reasoning
        if "reasoning" not in action:
            action["reasoning"] = "No reasoning provided"
        
        # Validate mutations if MUTATE_DAG
        if action.get("primary_op") == "MUTATE_DAG":
            mutations = action.get("mutations", [])
            
            if not mutations or not isinstance(mutations, list):
                logger.warning("MUTATE_DAG without valid mutations, using default")
                action["mutations"] = [{"type": "modify", "node_idx": -1}]
            
            # Validate each mutation
            valid_mutations = []
            for mut in mutations[:3]:  # Max 3 mutations
                if isinstance(mut, dict):
                    mut_type = mut.get("type", "modify")
                    if mut_type not in self.mutation_types:
                        mut_type = "modify"
                    
                    node_idx = mut.get("node_idx", 0)
                    try:
                        node_idx = int(node_idx)
                    except (TypeError, ValueError):
                        node_idx = 0
                    
                    valid_mutations.append({
                        "type": mut_type,
                        "node_idx": node_idx
                    })
            
            action["mutations"] = valid_mutations if valid_mutations else [{"type": "modify", "node_idx": 0}]
        
        return action

    def _log_dag_decision(self, action):
        """Log DAG-specific decision."""
        drop_str = f"DROP {action['drop_target']}" if action['drop_target'] else "NO DROP"
        primary_op = action.get('primary_op', 'NONE')
        
        if primary_op == "MUTATE_DAG":
            mutations = action.get('mutations', [])
            mut_str = ", ".join([f"{m['type']}@{m['node_idx']}" for m in mutations])
            print(f"\n🤖 Decision (conf={action['confidence']:.2f}): {drop_str}")
            print(f"   MUTATE_DAG target={action.get('primary_target')}: [{mut_str}]")
        else:
            target_str = f" {action.get('primary_target')}" if action.get('primary_target') else ""
            print(f"\n🤖 Decision (conf={action['confidence']:.2f}): {drop_str}, {primary_op}{target_str}")
        
        if action['confidence'] < 0.7 or primary_op == "MUTATE_DAG":
            print(f"   Reasoning: {action.get('reasoning', 'N/A')}")

    def _create_fallback_action(self, state):
        """Create a safe fallback action."""
        if state["population_size"] > 3:
            return {
                "drop_target": ["worst"],
                "primary_op": "CREATE",
                "primary_target": None,
                "confidence": 0.5,
                "reasoning": "Fallback: error recovery"
            }
        else:
            return {
                "drop_target": [],
                "primary_op": "CREATE",
                "primary_target": None,
                "confidence": 0.5,
                "reasoning": "Fallback: error recovery"
            }


class AdaptiveDAGMutationPolicy(DAGMutationPolicy):
    """DAG mutation policy with adaptive temperature."""
    
    def decide(self, state, search_space=None, save_dir=None, task_description=None):
        """Decide with adaptive temperature."""
        progress = state['step'] / max(state['max_steps'], 1)
        
        # More exploration early, more refinement late
        if progress < 0.3:
            self.temperature = 0.8  # High exploration
        elif progress < 0.7:
            self.temperature = 0.5  # Balanced
        else:
            self.temperature = 0.2  # Focused refinement
        
        print(f"🌡️ T={self.temperature:.1f}")
        return super().decide(state, search_space, save_dir, task_description)

# Utility function to apply LLM-specified mutations
def apply_llm_mutations(array_value, mutations, array_var):
    """
    Apply LLM-specified mutations to an ArrayVar containing an EvoDagVariable.

    Parameters
    ----------
    array_value : list
        Value of the ArrayVar (contains the DAG)
    mutations : list[dict]
        [{"type": str, "node_idx": int}, ...]
    array_var : ArrayVar
        The ArrayVar definition (search space)

    Returns
    -------
    list
        Mutated Array value
    """

    # Deep copy of array value
    mutated_array = copy.deepcopy(array_value)

    # Find the EvoDagVariable inside the ArrayVar
    dag_var = None
    dag_idx = None
    for i, v in enumerate(array_var.values):
        if isinstance(v, EvoDagVariable):
            dag_var = v
            dag_idx = i
            break

    if dag_var is None:
        raise ValueError("No EvoDagVariable found in ArrayVar")

    # Get DAG value and its neighborhood
    dag_value = mutated_array[dag_idx]
    dag_neighbor = dag_var.neighbor  # EvoDagInterval

    for mutation in mutations:
        mut_type = mutation.get("type", "modify")
        node_idx = mutation.get("node_idx", 0)

        # Handle negative indices
        if node_idx < 0:
            node_idx = len(dag_value.operations) + node_idx

        # Clamp
        node_idx = max(0, min(node_idx, len(dag_value.operations) - 1))

        try:
            dag_value = dag_neighbor.modification(
                mut_type,
                node_idx,
                dag_value
            )
        except Exception as e:
            logger.error(f"Error applying mutation {mutation}: {e}")

    # Put back mutated DAG
    mutated_array[dag_idx] = dag_value
    return mutated_array


class DAGAgenticSearchAlgorithm(AgenticSearchAlgorithm):
    """
    Agentic search with LLM control over DAG mutations.
    
    Extends AgenticSearchAlgorithm to handle MUTATE_DAG operations
    where the LLM specifies exactly which mutations to apply.
    """
    
    def _execute_primary_operation(self, action, state):
        """Execute primary operation with DAG mutation support."""
        primary_op = action.get("primary_op")
        primary_target = action.get("primary_target")

        if not primary_op or primary_op not in ["CREATE", "MUTATE", "MUTATE_DAG", "TRAIN"]:
            return []

        if primary_op == "CREATE":
            return self._create_individual()
        
        elif primary_op == "MUTATE_DAG":
            target_idx = self._resolve_target(primary_target, state)
            mutations = action.get("mutations", [])
            return self._mutate_dag_individual(target_idx, mutations)
        
        elif primary_op == "MUTATE":
            # Standard mutation (uses search_space.neighbor)
            target_idx = self._resolve_target(primary_target, state)
            return self._mutate_individual(target_idx)
        
        elif primary_op == "TRAIN":
            target_idx = self._resolve_target(primary_target, state)
            return self._train_individual(target_idx)
        
        return []

    def _mutate_dag_individual(self, parent_idx, mutations):
        """
        Create a mutant from parent with LLM-specified mutations.
        
        Parameters
        ----------
        parent_idx : int
            Index of parent configuration
        mutations : list[dict]
            List of specific mutations to apply
            Format: [{"type": str, "node_idx": int}, ...]
        
        Returns
        -------
        list[int]
            List containing new individual index
        """
        try:
            # Load parent configuration
            with open(f"{self.save_dir}/x_{parent_idx}.pkl", 'rb') as f:
                parent_config = pickle.load(f)
        except FileNotFoundError:
            logger.warning(f"Parent config {parent_idx} not found, creating random instead")
            return self._create_individual()
        
        try:
            # Apply LLM-specified mutations
            mutant_config = self._apply_targeted_mutations(parent_config, mutations)
            
            # Save mutant
            new_idx = self.K
            self.K += 1
            
            with open(f"{self.save_dir}/x_{new_idx}.pkl", 'wb') as f:
                pickle.dump(mutant_config, f)
            
            self.sent[new_idx] = {"N": 0, "N_bar": 0, "UCBLoss": 0}
            
            mut_str = ", ".join([f"{m['type']}@{m.get('node_idx', '?')}" for m in mutations])
            print(f"🧬 DAG mutation: idx={parent_idx} → new idx={new_idx}")
            print(f"   Applied: [{mut_str}]")
            
            return [new_idx]
            
        except Exception as e:
            logger.error(f"Error in targeted DAG mutation: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to standard mutation
            logger.info("Falling back to standard mutation")
            return self._mutate_individual(parent_idx)

    def _apply_targeted_mutations(self, config, mutations):
        """
        Apply specific mutations to a DAG inside an ArrayVar configuration.
        """

        # Deep copy of Array value
        mutated = copy.deepcopy(config)

        # 1️⃣ Trouver le DAG dans l'ArrayVar
        dag_var = None
        dag_idx = None
        for i, var in enumerate(self.search_space.values):
            if isinstance(var, EvoDagVariable):
                dag_var = var
                dag_idx = i
                break

        if dag_var is None:
            logger.warning("No EvoDagVariable found, fallback to standard neighbor")
            return self.search_space.neighbor(config)

        dag_value = mutated[dag_idx]
        dag_neighbor = dag_var.neighbor  # EvoDagInterval

        # 2️⃣ Appliquer les mutations ciblées
        for mutation in mutations:
            mut_type = mutation.get("type", "modify")
            node_idx = mutation.get("node_idx", 0)

            # Indices négatifs
            if node_idx < 0:
                node_idx = len(dag_value.operations) + node_idx

            # Clamp
            node_idx = max(0, min(node_idx, len(dag_value.operations) - 1))

            try:
                dag_value = dag_neighbor.modification(
                    mut_type,
                    node_idx,
                    dag_value
                )
                logger.debug(f"Applied {mut_type} on DAG node {node_idx}")
            except Exception as e:
                logger.error(f"Error applying {mut_type} at node {node_idx}: {e}")
                continue

        # 3️⃣ Validation du DAG
        try:
            dag_value.assert_adj_matrix()
        except AssertionError as e:
            logger.error(f"Invalid DAG after mutations, fallback: {e}")
            return self.search_space.neighbor(config)

        # 4️⃣ Réinjecter le DAG muté
        mutated[dag_idx] = dag_value
        return mutated