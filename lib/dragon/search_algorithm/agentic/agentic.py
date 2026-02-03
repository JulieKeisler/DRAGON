"""
Agentic Search Algorithm
========================
An evolutionary search algorithm controlled by AI agents (LLM or heuristic-based).
Extends Mutant-UCB with intelligent decision-making for selecting operations (TRAIN/MUTATE/DROP/CREATE).
"""

from dragon.search_algorithm.mutant_ucb import Mutant_UCB
from dragon.utils.tools import logger
from abc import ABC, abstractmethod
import pickle
import numpy as np
import pandas as pd
from groq import Groq
import json
from enum import Enum


class ActionType(Enum):
    """Available operations for modifying the population."""
    TRAIN = "TRAIN"      # Continue training an individual
    MUTATE = "MUTATE"    # Create a new individual by mutation
    DROP = "DROP"        # Remove an individual (required every iteration)
    CREATE = "CREATE"    # Create a completely new random individual


class AgentPolicy(ABC):
    """
    Abstract base class for decision-making policies.
    
    A policy decides which actions to take and which individuals to target.
    Must return DROP targets plus optionally a primary action (TRAIN/MUTATE/CREATE).
    """
    
    @abstractmethod
    def decide(self, state, search_space=None, save_dir=None, task_description=None) -> dict:
        """
        Decide the next actions based on current state.
        
        Parameters
        ----------
        state : dict
            Current population state with keys: step, max_steps, population_size, best, worst, full_population, stats
        search_space : optional
            Search space object for creating configurations
        save_dir : str, optional
            Directory where configurations are saved
        task_description : str, optional
            Description of the optimization task
            
        Returns
        -------
        dict
            Actions with keys:
            - drop_target: list[str|int] - targets to drop (required, can be empty list)
            - primary_op: "TRAIN" | "MUTATE" | "CREATE" | None
            - primary_target: target for primary operation (not needed for CREATE)
            - confidence: float in [0, 1]
            - reasoning: str (brief explanation)
        """
        pass


class HeuristicAgentPolicy(AgentPolicy):
    """
    Simple rule-based policy using heuristics.
    
    Parameters
    ----------
    stagnation_threshold : int, default=5
        Number of training iterations before considering an individual stagnant
    variance_threshold : float, default=0.01
        Loss variance threshold for triggering exploration
    min_population_size : int, default=3
        Minimum population size before triggering CREATE
    allow_train : bool, default=True
        Whether TRAIN operation is allowed (set False if N=1)
    """
    
    def __init__(self, stagnation_threshold=5, variance_threshold=0.01, 
                 min_population_size=3, allow_train=True):
        self.stagnation_threshold = stagnation_threshold
        self.variance_threshold = variance_threshold
        self.min_population_size = min_population_size
        self.allow_train = allow_train

    def decide(self, state, search_space=None, save_dir=None, task_description=None):
        best = state["best"]
        std_loss = state["stats"]["std_loss"]
        pop_size = state["population_size"]

        # Drop worst if population is large enough
        drop_target = ["worst"] if pop_size > self.min_population_size else []

        # Decide primary action
        if pop_size < self.min_population_size:
            primary_op = "CREATE"
            primary_target = None
            confidence = 0.95
            reasoning = "Population too small, creating new individual"
        elif std_loss > self.variance_threshold:
            primary_op = "MUTATE"
            primary_target = "worst"
            confidence = 0.9
            reasoning = "High variance, exploring worst individual"
        elif self.allow_train and best["N"] < self.stagnation_threshold:
            primary_op = "TRAIN"
            primary_target = "best"
            confidence = 0.95
            reasoning = "Exploiting best individual"
        else:
            primary_op = "MUTATE"
            primary_target = "best"
            confidence = 0.8
            reasoning = "Mutating best for local exploration"

        return {
            "drop_target": drop_target,
            "primary_op": primary_op,
            "primary_target": primary_target,
            "confidence": confidence,
            "reasoning": reasoning
        }


class EnhancedLLMPolicy(AgentPolicy):
    """
    Advanced LLM policy with memory and self-reflection.
    
    Features:
    - Maintains history of actions and outcomes
    - Analyzes past performance to improve decisions
    - Self-calibrates confidence based on uncertainty
    - Accesses actual configurations for richer context
    
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
        Whether TRAIN is allowed (set False if N=1)
    task_description : str, optional
        Task description for context
    """
    
    def __init__(self, model="llama-3.3-70b-versatile", temperature=0.3, api_key=None, 
                 history_length=10, allow_train=True, task_description=None):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.history = []
        self.history_length = history_length
        self.action_outcomes = []
        self.allow_train = allow_train
        self.task_description = task_description

    def decide(self, state, search_space=None, save_dir=None, task_description=None):
        """Make an intelligent decision with rich context and self-analysis."""
        
        task_desc = task_description or self.task_description or "optimization"
        ops_available = ["MUTATE", "CREATE"] + (["TRAIN"] if self.allow_train else [])

        enriched_pop = self._enrich_population(state, save_dir)
        trends = self._compute_trends()
        performance_analysis = self._analyze_past_performance()
        observations = self._get_critical_observations(state)

        # Create JSON-safe population display
        pop_display = self._create_safe_population_display(enriched_pop)

        prompt = self._build_prompt(
            task_desc, state, pop_display, trends, 
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
            action = self._validate_and_fix_action(action, ops_available, state)
            
            # Log decision
            self._log_decision(action)
            self._add_to_history(state, action)
            
            return action

        except Exception as e:
            logger.error(f"LLM error: {e}")
            import traceback
            traceback.print_exc()
            
            fallback = self._create_fallback_action(state)
            self._add_to_history(state, fallback)
            return fallback

    def _build_prompt(self, task_desc, state, pop_display, trends, performance, observations, ops_available):
        """Build the LLM prompt."""
        return f"""You are an expert AI controlling an evolutionary search algorithm for {task_desc}.

📊 STATUS (Step {state['step']}/{state['max_steps']}, {(state['step']/state['max_steps']*100):.0f}%):
Best: idx={state['best']['idx']}, loss={state['best']['loss']:.6f}, N={state['best']['N']}
Worst: idx={state['worst']['idx']}, loss={state['worst']['loss']:.6f}, N={state['worst']['N']}
Stats: μ={state['stats']['mean_loss']:.6f}, σ={state['stats']['std_loss']:.6f}
Population: {state['population_size']} individuals

Population sample:
{json.dumps(pop_display[:3], indent=2)}

History: {self._format_history()}
Trends: {trends}
Performance: {performance}
{f"⚠️ {observations}" if observations != "No critical issues." else ""}

TASK:
1. DROP: Select 0-2 individuals to remove (empty list = keep all)
2. PRIMARY: Choose ONE operation from {ops_available} or null

Operations:
{f"- TRAIN: Continue training (increases N)" if self.allow_train else ""}
- MUTATE: Create new via mutation of selected individual
- CREATE: Add completely random individual

Strategy:
- Early (<40%): Explore (CREATE/MUTATE diverse targets)
- Mid (40-90%): Balance exploitation/exploration
- Late (>90%): Exploit ({"TRAIN/MUTATE best" if self.allow_train else "MUTATE best"})
- Low variance (<0.01): Need diversity (CREATE/MUTATE)
- Overfitting (N big): Explore alternatives

Return ONLY JSON:
{{
    "drop_target": ["worst"] | [<int>] | [<int>, <int>] | [<int>, <int>, <int>],
    "primary_op": "{'" | "'.join(ops_available)}" | null,
    "primary_target": "best" | "worst" | "random" | <int> | null,
    "confidence": <float 0-1>,
    "reasoning": "<brief explanation; explain confidence>"
}}"""

    def _create_safe_population_display(self, enriched_pop):
        """Convert population to JSON-safe format."""
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
        return pop_display

    def _validate_and_fix_action(self, action, ops_available, state):
        """Validate and fix LLM action to ensure it's valid."""
        
        # Ensure drop_target is a list
        if "drop_target" not in action or action["drop_target"] is None:
            action["drop_target"] = ["worst"]
        elif not isinstance(action["drop_target"], list):
            action["drop_target"] = [action["drop_target"]]
        
        # Validate and fix primary_op
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
        
        # Ensure reasoning exists
        if "reasoning" not in action:
            action["reasoning"] = "No reasoning provided"
        
        return action

    def _create_fallback_action(self, state):
        """Create a safe fallback action."""
        # Drop worst if population large enough, otherwise create
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

    def _log_decision(self, action):
        """Log the decision concisely."""
        drop_str = f"DROP {action['drop_target']}" if action['drop_target'] else "NO DROP"
        primary_str = f"{action.get('primary_op', 'NONE')}"
        if action.get('primary_target'):
            primary_str += f" {action['primary_target']}"
        
        print(f"\n🤖 Decision (conf={action['confidence']:.2f}): {drop_str}, {primary_str}")
        
        if action['confidence'] < 0.7:
            print(f"   Reasoning: {action['reasoning']}")

    def update_outcome(self, action_idx, loss_after):
        """Update action outcome after evaluation."""
        if action_idx < len(self.history):
            entry = self.history[action_idx]
            entry['loss_after'] = loss_after
            
            improvement = entry['best_loss_before'] - loss_after
            entry['improvement'] = improvement
            entry['improved'] = improvement > 0
            
            self.action_outcomes.append({
                'op': entry['action'].get('primary_op'),
                'drop': entry['action']['drop_target'],
                'confidence': entry['action']['confidence'],
                'improved': entry['improved'],
                'improvement': improvement
            })
    
    def _enrich_population(self, state, save_dir):
        """Add configuration details (robust version)."""
        enriched_pop = []
        
        for ind in state['full_population']:
            ind_copy = ind.copy()
            
            if save_dir:
                try:
                    with open(f"{save_dir}/x_{ind['idx']}.pkl", 'rb') as f:
                        config = pickle.load(f)
                    config_str = str(config)#[:100]
                    #if len(str(config)) > 100:
                    #    config_str += "..."
                    ind_copy['config_preview'] = config_str
                except Exception:
                    ind_copy['config_preview'] = "N/A"
            
            enriched_pop.append(ind_copy)
        
        return enriched_pop
    
    def _add_to_history(self, state, action):
        """Record action with context."""
        self.history.append({
            "step": state['step'],
            "action": action,
            "best_loss_before": state['best']['loss'],
            "pop_size": state['population_size'],
            "std_loss": state['stats']['std_loss']
        })
        if len(self.history) > self.history_length:
            self.history.pop(0)
    
    def _analyze_past_performance(self):
        """Analyze performance (succinct)."""
        if len(self.action_outcomes) < 1:
            return "Insufficient data"
        
        recent = self.action_outcomes[-10:]
        ops_perf = {}
        for o in recent:
            op = o['op'] or 'NONE'
            if op not in ops_perf:
                ops_perf[op] = {'cnt': 0, 'ok': 0}
            ops_perf[op]['cnt'] += 1
            if o['improved']:
                ops_perf[op]['ok'] += 1
        
        parts = [f"{op}:{s['ok']}/{s['cnt']}" for op, s in ops_perf.items()]
        return ", ".join(parts)
    
    def _get_critical_observations(self, state):
        """Identify critical issues (brief)."""
        obs = []
        
        if state['best']['N'] > 10:
            obs.append(f"Best overtrained (N={state['best']['N']})")
        if state['stats']['std_loss'] < 0.01:
            obs.append("Low variance - needs diversity")
        
        progress = state['step'] / state['max_steps']
        if progress > 0.9:
            obs.append("Late stage - focus exploitation")
        elif state['population_size'] < 4 and progress < 0.5:
            obs.append(f"Small population ({state['population_size']})")
        
        if len(self.history) >= 3:
            recent = [h['best_loss_before'] for h in self.history[-3:]]
            if recent[-1] > recent[0]:
                obs.append("Loss worsening")
        
        return "; ".join(obs) if obs else "No critical issues."
    
    def _format_history(self):
        """Format history (compact)."""
        if not self.history:
            return "None"
        
        parts = []
        for h in self.history[-3:]:
            op = h['action'].get('primary_op', 'NONE')
            parts.append(f"#{h['step']}:{op}")
        return ", ".join(parts)
    
    def _compute_trends(self):
        """Compute trends (succinct)."""
        if len(self.history) < 3:
            return "Insufficient history"
        
        recent = self.history[-5:]
        losses = [h['best_loss_before'] for h in recent]
        trend = "↓" if losses[-1] < losses[0] else "↑"
        change = ((losses[-1] - losses[0]) / losses[0] * 100) if losses[0] != 0 else 0
        
        ops = [h['action'].get('primary_op', 'NONE') for h in recent]
        op_counts = {op: ops.count(op) for op in set(ops)}
        
        pop_sizes = [h['pop_size'] for h in recent]
        pop_delta = pop_sizes[-1] - pop_sizes[0]
        
        return f"Loss{trend}{change:+.1f}%, Ops={op_counts}, Pop{pop_delta:+d}"


class AdaptiveLLMPolicy(EnhancedLLMPolicy):
    """
    Enhanced policy with adaptive temperature control.
    
    Automatically adjusts temperature based on search progress:
    - Early: High temperature (exploration)
    - Mid: Medium temperature (balanced)
    - Late: Low temperature (exploitation)
    """
    
    def __init__(self, model="llama-3.3-70b-versatile", api_key=None, history_length=10, 
                 allow_train=True, task_description=None):
        super().__init__(model=model, temperature=0.3, api_key=api_key, 
                        history_length=history_length, allow_train=allow_train,
                        task_description=task_description)
        
    def decide(self, state, search_space=None, save_dir=None, task_description=None):
        """Decide with adaptive temperature."""
        progress = state['step'] / max(state['max_steps'], 1)  # Avoid division by zero
        
        if progress < 0.3:
            self.temperature = 0.7
        elif progress < 0.7:
            self.temperature = 0.4
        else:
            self.temperature = 0.1
        
        print(f"🌡️ T={self.temperature:.1f}")
        return super().decide(state, search_space, save_dir, task_description)


class AgenticSearchAlgorithm(Mutant_UCB):
    """
    Evolutionary search with AI agent control.
    
    Replaces UCB selection with intelligent agent decision-making for operation selection.
    
    Parameters
    ----------
    search_space : SearchSpace
        Search space to explore
    T : int
        Training epochs per evaluation
    K : int
        Initial population size
    N : int
        Evaluations per config (if N=1, TRAIN disabled)
    E : int
        Not used in this implementation
    evaluation : callable
        Function to evaluate configurations
    save_dir : str
        Directory to save results
    policy : AgentPolicy (required)
        Decision-making policy
    task_description : str, optional
        Task description for LLM context
    **args
        Additional arguments for parent class
    """
    
    def __init__(self, search_space, T, K, N, E, evaluation, save_dir, 
                 models=None, pop_path=None, verbose=False, policy=None, 
                 task_description=None, clean_all=True, **args):
        super().__init__(search_space, T, K, N, E, evaluation, save_dir, 
                        models, pop_path, verbose, clean_all, **args)
        
        if policy is None:
            raise ValueError("policy parameter is required")
        
        # Disable TRAIN if N=1 (only one evaluation per config)
        if N == 1 and hasattr(policy, 'allow_train'):
            policy.allow_train = False
            logger.info("TRAIN operation disabled (N=1)")
        
        self.agent_policy = policy
        self.step_counter = 0
        self.last_action_idx = 0
        self.task_description = task_description

    def extract_state(self):
        """
        Extract current population state for policy decision-making.
        
        Returns
        -------
        dict or None
            State dictionary with all necessary information, or None if population is empty
        """
        if len(self.storage) == 0:
            return None

        # Extract losses and training counts, converting to native Python types
        losses = {i: float(v["UCBLoss"]) for i, v in self.storage.items()}
        Ns = {i: int(v["N"]) for i, v in self.storage.items()}
        
        best_idx = min(losses, key=losses.get)
        worst_idx = max(losses, key=losses.get)

        return {
            "step": self.step_counter,
            "max_steps": max(self.n_iterations - len(self.storage), 1),  # Avoid zero
            "population_size": len(self.storage),
            "best": {
                "idx": int(best_idx), 
                "loss": float(losses[best_idx]), 
                "N": int(Ns[best_idx])
            },
            "worst": {
                "idx": int(worst_idx), 
                "loss": float(losses[worst_idx]), 
                "N": int(Ns[worst_idx])
            },
            "full_population": [
                {"idx": int(i), "loss": float(losses[i]), "N": int(Ns[i])} 
                for i in sorted(self.storage.keys())
            ],
            "stats": {
                "mean_loss": float(np.mean(list(losses.values()))),
                "std_loss": float(np.std(list(losses.values()))),
                "mean_N": float(np.mean(list(Ns.values()))),
                "std_N": float(np.std(list(Ns.values())))
            }
        }

    def select_next_configurations(self):
        """
        Select next configuration(s) to evaluate using agent policy.
        
        Returns
        -------
        list
            List of configuration indices to evaluate
        """
        state = self.extract_state()
        
        # Initialize population if empty
        if state is None or len(self.storage) == 0:
            logger.info("Initializing population...")
            self.create_population()
            for idx in range(self.population_size):
                loss, idx = self.evaluate(idx, self.time)
                self.save_best_model(idx, loss)
            state = self.extract_state()

        self.step_counter += 1
        
        # Get decision from policy
        action = self.agent_policy.decide(
            state,
            search_space=self.search_space,
            save_dir=self.save_dir,
            task_description=self.task_description
        )
        
        self.last_action_idx = len(self.agent_policy.history) - 1

        # Execute primary operation first (creates new individuals)
        new_idxs = self._execute_primary_operation(action, state)

        # Then apply drops (removes individuals)
        self._execute_drops(action, state)

        return new_idxs

    def _execute_primary_operation(self, action, state):
        """Execute the primary operation (CREATE/MUTATE/TRAIN)."""
        primary_op = action.get("primary_op")
        primary_target = action.get("primary_target")

        if not primary_op or primary_op not in ["CREATE", "MUTATE", "TRAIN"]:
            return []

        if primary_op == "CREATE":
            return self._create_individual()
        
        elif primary_op == "MUTATE":
            target_idx = self._resolve_target(primary_target, state)
            return self._mutate_individual(target_idx)
        
        elif primary_op == "TRAIN":
            target_idx = self._resolve_target(primary_target, state)
            return self._train_individual(target_idx)
        
        return []

    def _create_individual(self):
        """Create a new random individual."""
        x = self.search_space.random()
        new_idx = self.K
        self.K += 1
        
        with open(f"{self.save_dir}/x_{new_idx}.pkl", 'wb') as f:
            pickle.dump(x, f)
        
        self.sent[new_idx] = {"N": 0, "N_bar": 0, "UCBLoss": 0}
        print(f"✨ Created new individual idx={new_idx}")
        return [new_idx]

    def _mutate_individual(self, parent_idx):
        """Create a mutant from a parent individual."""
        try:
            with open(f"{self.save_dir}/x_{parent_idx}.pkl", 'rb') as f:
                parent_config = pickle.load(f)
        except FileNotFoundError:
            logger.warning(f"Parent config {parent_idx} not found, creating random instead")
            return self._create_individual()
        
        mutant_config = self.search_space.neighbor(parent_config)
        new_idx = self.K
        self.K += 1
        
        with open(f"{self.save_dir}/x_{new_idx}.pkl", 'wb') as f:
            pickle.dump(mutant_config, f)
        
        self.sent[new_idx] = {"N": 0, "N_bar": 0, "UCBLoss": 0}
        print(f"🔄 Mutated idx={parent_idx} → new idx={new_idx}")
        return [new_idx]

    def _train_individual(self, target_idx):
        """Continue training an existing individual."""
        if target_idx in self.storage:
            self.sent[target_idx] = self.storage.pop(target_idx)
            print(f"🎓 Training idx={target_idx}")
            return [target_idx]
        else:
            logger.warning(f"TRAIN target {target_idx} not in storage, skipping")
            return []

    def _execute_drops(self, action, state):
        """Execute DROP operations on multiple targets."""
        drop_targets = action.get("drop_target", [])
        
        if not isinstance(drop_targets, list):
            drop_targets = [drop_targets]

        for dt in drop_targets:
            idx_to_drop = self._resolve_target(dt, state)
            if idx_to_drop in self.storage:
                self.storage.pop(idx_to_drop)
                print(f"🗑️ Dropped idx={idx_to_drop}")
            else:
                logger.debug(f"Drop target {idx_to_drop} not in storage")

    def _resolve_target(self, target, state):
        """
        Resolve target string to actual index.
        
        Parameters
        ----------
        target : str or int
            Target specification ("best", "worst", "random", or integer index)
        state : dict
            Current state
            
        Returns
        -------
        int
            Resolved index
        """
        if target == "best":
            return state["best"]["idx"]
        elif target == "worst":
            return state["worst"]["idx"]
        elif target == "random":
            if len(self.storage) > 0:
                return int(np.random.choice(list(self.storage.keys())))
            else:
                logger.warning("Cannot select random from empty storage")
                return 0
        else:
            return int(target)
    
    def process_evaluated_configuration(self, idx, loss):
        """
        Process evaluated configuration and provide feedback to policy.
        
        Parameters
        ----------
        idx : int
            Configuration index
        loss : float
            Evaluation loss
            
        Returns
        -------
        tuple
            (delete_flag, row_dataframe, loss)
        """
        # Update storage with evaluation results
        if idx in self.sent:
            self.sent[idx]['Idx'] = idx
            self.sent[idx]['Loss'] = loss
            self.sent[idx]['UCBLoss'] = (
                loss + self.sent[idx]['N_bar'] * self.sent[idx]['UCBLoss']
            ) / (self.sent[idx]['N_bar'] + 1)
            self.sent[idx]['N'] += 1
            self.sent[idx]['N_bar'] += 1
            self.storage[idx] = self.sent.pop(idx)
        else:
            # Handle initial population evaluation
            self.storage[idx] = {
                "Idx": idx,
                "N": 1, 
                "N_bar": 1, 
                "UCBLoss": loss, 
                "Loss": loss
            }
        
        # Provide feedback to policy for learning
        if hasattr(self.agent_policy, 'update_outcome'):
            self.agent_policy.update_outcome(self.last_action_idx, loss)
        
        # Return results in expected format
        return False, pd.DataFrame({k: [v] for k, v in self.storage[idx].items()}), loss
    
    def process_evaluated_row(self, row):
        """
        Reconstruct storage from saved computation file.
        
        Parameters
        ----------
        row : dict
            Row from computation_file.csv
        """
        loss = row['UCBLoss']
        self.storage[row['Idx']] = {
            "Loss": row['Loss'], 
            "N": row['N'], 
            "N_bar": row['N_bar'], 
            "UCBLoss": loss
        }
        
        if self.min_loss > loss:
            logger.info(f'Best found! {loss:.6f} < {self.min_loss:.6f}')
            self.min_loss = loss