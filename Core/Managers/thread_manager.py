
import threading
import queue
import traceback
from typing import Any, Callable, Dict, Optional

# ============================================================
#  WORKER
#
#  Runs a long‑running task in a background thread.
#  Communicates with the UI via a thread‑safe queue.
# ============================================================
class Worker:

    def __init__(self, task_id: str, target: Callable[..., Any], args: tuple = (), kwargs: Optional[dict] = None):
      
        self.task_id = task_id
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}

        self.queue: queue.Queue = queue.Queue()
        self.thread: Optional[threading.Thread] = None
        self.cancel_flag: bool = False

    # ------------------------------------------------------------
    # Thread lifecycle
    # Start the worker thread.
    # ------------------------------------------------------------
    def start(self):
     
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------
    # Internal execution wrapper
    # ------------------------------------------------------------
    def cancel(self):

        self.cancel_flag = True
        self.queue.put(("status", "Cancellation requested. Waiting for task to stop safely..."))

    # ------------------------------------------------------------
    # Internal execution wrapper
    # ------------------------------------------------------------
    def _run(self):
        
        try:
            self.queue.put(("status", f"Task {self.task_id} started"))

            # Run the target function
            result = self.target(
                *self.args,
                **self.kwargs,
                worker=self,  # pass worker for progress updates
            )

            if not self.cancel_flag:
                self.queue.put(("result", result))

        except Exception:
            tb = traceback.format_exc()
            self.queue.put(("error", tb))

        finally:
            self.queue.put(("finished", None))

    # ------------------------------------------------------------
    # Helper for tasks to send progress updates
    # Tasks can call worker.progress('msg') to update UI.
    # ------------------------------------------------------------
    def progress(self, message: str):        
        self.queue.put(("progress", message))

    def status(self, message: str):
        self.queue.put(("status", message))


# ============================================================
#  TASK MANAGER
#
#    Manages multiple background tasks.
#    Polls messages from workers and cleans up finished tasks.
# ============================================================
class TaskManager:

    def __init__(self):
        self.tasks: Dict[str, Worker] = {}

    # ------------------------------------------------------------
    # Start a new task
    # ------------------------------------------------------------
    def start_task(self, task_id: str, target: Callable[..., Any], *args, **kwargs) -> Worker:

        if task_id in self.tasks:
            raise RuntimeError(f"Task '{task_id}' is already running")

        worker = Worker(task_id, target, args, kwargs)
        self.tasks[task_id] = worker
        worker.start()
        return worker

    # ------------------------------------------------------------
    # Poll messages from all workers
    # Returns a list of (task_id, msg_type, data) tuples.
    # Removes finished tasks automatically.
    # ------------------------------------------------------------
    def get_messages(self):

        messages = []

        for task_id, worker in list(self.tasks.items()):

            try:

                while True:

                    msg_type, data = worker.queue.get_nowait()
                    messages.append((task_id, msg_type, data))

                    if msg_type == "finished":
                        del self.tasks[task_id]
                        break

            except queue.Empty:
                pass

        return messages

    # ------------------------------------------------------------
    # Cancel a running task
    # ------------------------------------------------------------
    def cancel_task(self, task_id: str):

        if task_id in self.tasks:
            self.tasks[task_id].cancel()
