# Lab 2 - Smart Classroom Agent
# This program checks temperature and room occupancy.

class ClassroomAgent:
    def act(self, temp, occupied):
        if not occupied:
            return "ECO"
        if temp > 26:
            return "COOL"
        if temp < 20:
            return "WARM"
        return "IDLE"


agent = ClassroomAgent()

percepts = [(29, True), (29, True), (20, True),
            (26, True), (19, True), (29, False)]

print("Step | Percept | Target mode")
print("-----------------------------")

for step, percept in enumerate(percepts, 1):
    print(step, percept, "|", agent.act(*percept))
