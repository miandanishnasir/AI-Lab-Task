# Lab 2 - Edge Case and Boundary Check
# This program checks exact temperature limits and noisy readings.

class ClassroomAgent:
    def __init__(self):
        self.previous_mode = "NONE"

    def act(self, temp, occupied):
        if not occupied:
            target_mode = "ECO"
        elif temp > 26:
            target_mode = "COOL"
        elif temp < 20:
            target_mode = "WARM"
        else:
            target_mode = "IDLE"

        if target_mode == self.previous_mode:
            send_command = False
        else:
            send_command = True

        old_mode = self.previous_mode
        self.previous_mode = target_mode
        return old_mode, target_mode, send_command


agent = ClassroomAgent()

# 20 and 26 are exact boundaries.
# 25.9, 26.1 and 25.8 show small changes around 26.
test_percepts = [(20, True), (26, True), (25.9, True),
                 (26.1, True), (25.8, True), (24, False)]

print("Previous | Percept | Target | Command")
print("--------------------------------------")

for percept in test_percepts:
    previous, target, send = agent.act(*percept)
    if send:
        command = "SEND"
    else:
        command = "NO COMMAND"
    print(previous, "|", percept, "|", target, "|", command)
