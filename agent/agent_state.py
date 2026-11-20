import time
import uuid # For generating unique IDs if needed for tasks

# Assuming AgentState is in the same package or path is adjusted
# from .agent_loop.agent_state import AgentState
# For now, we'll assume AgentState is passed and has task_queue attribute

class AgentState:
    def __init__(self, config=None):
        self.conversation_history = []
        self.context_buffer = {}
        self.task_queue = []
        self.current_task = None
        self.task_results = {}
        self.config = config or {}
        
        self.flipper_agent = None
        self.llm_agent = None
        self.app_instance = None

        self.awaiting_human_input = False
        self.human_input_request = None # Stores details of the human input request
        
    def add_to_history(self, entry):
        """Add entry to conversation history, pruning oldest entries to stay
        within the configured token budget.

        Smaller prompts mean faster LLM calls; on long multi-step tasks an
        unbounded history would keep growing every reflection round and slow
        each call down. Tokens are approximated as characters / 4.
        """
        self.conversation_history.append(entry)

        max_tokens = self.config.get("llm", {}).get("max_history_tokens", 6000)
        char_budget = max(1000, max_tokens * 4)
        total = sum(len(str(e.get("content", ""))) for e in self.conversation_history)
        # Keep at least the most recent entry even if it alone exceeds the budget.
        while len(self.conversation_history) > 1 and total > char_budget:
            removed = self.conversation_history.pop(0)
            total -= len(str(removed.get("content", "")))
        
    def update_context(self, key, value):
        """Update a value in the context buffer."""
        self.context_buffer[key] = value
        
    def clear_for_new_session(self):
        """Reset state for a new user session/request."""
        self.conversation_history = []
        # self.context_buffer = {} # Decide if context buffer should be fully cleared
        self.task_queue = []
        self.current_task = None
        self.task_results = {}
        self.awaiting_human_input = False
        self.human_input_request = None
        
    def request_human_input(self, request_type, prompt, options=None):
        """Set state to await human input with given prompt."""
        self.awaiting_human_input = True
        self.human_input_request = {
            "type": request_type,  # "question", "approval", "selection", etc.
            "prompt": prompt,
            "options": options,    # Optional list of choices
            "timestamp": time.time()
        }
        
    def process_human_input(self, input_value):
        """Process received human input and update state."""
        if not self.awaiting_human_input or self.human_input_request is None:
            # Log or handle error: received human input when not expecting it
            return None 

        self.awaiting_human_input = False
        
        history_entry = {
            "role": "human",
            "content": f"Response to '{self.human_input_request['type']}': {input_value}"
        }
        self.add_to_history(history_entry)
        
        result = {"request": self.human_input_request, "response": input_value}
        self.human_input_request = None # Clear the request
        return result

    def get_full_context_for_llm(self):
        """Prepares the full context including history and current state for LLM."""
        context = self.conversation_history.copy()
        if self.current_task:
            context.append({"role": "system", "content": f"Current task: {self.current_task}"})
        if self.task_queue:
            task_list_str = []
            for task in self.task_queue[:3]:
                task_list_str.append(f"- {task.get('type')}: {task.get('parameters', {})}")
            
            if len(self.task_queue) > 3:
                task_list_str.append(f"...and {len(self.task_queue) - 3} more task(s).")
                
            context.append({"role": "system", "content": f"Remaining tasks in queue:\n" + "\n".join(task_list_str)})
        return context