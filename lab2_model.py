# Lab 2 - Stateful Smart Classroom Agent
# This agent remembers its previous mode.
# It sends a command only when the target mode changes.

class ClassroomAgent:
    def __init__(self):
        self.previous_mode = "NONE"

    def act(self, temp, occupied):
        # Decide the target mode using the lab rules.
        if not occupied:
            target_mode = "ECO"
        elif temp > 26:
            target_mode = "COOL"
        elif temp < 20:
            target_mode = "WARM"
        else:
            target_mode = "IDLE"

        # Check if a new command is needed.
        if target_mode == self.previous_mode:
            send_command = False
        else:
            send_command = True

        old_mode = self.previous_mode
        self.previous_mode = target_mode

        return old_mode, target_mode, send_command


agent = ClassroomAgent()

# Includes repeated readings, exact boundaries, a mode change,
# and an empty classroom.
percepts = [(29, True), (29, True), (20, True),
            (26, True), (19, True), (29, False)]

print("Step | Previous mode | Percept | Target mode | Command?")
print("-------------------------------------------------------")

for step, percept in enumerate(percepts, 1):
    previous, target, send = agent.act(*percept)

    if send:
        command = "SEND"
    else:
        command = "NO COMMAND"

    print(step, "|", previous, "|", percept, "|", target, "|", command)
