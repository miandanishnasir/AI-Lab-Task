#AI lab task

print("PART 1: Rule-Based vs Probability-Based AI")
print("-------------------------------------------")

score = 82
documents_complete = True
probability = 0.81

if documents_complete == True and score >= 70:
    rule_result = "Review"
else:
    rule_result = "Hold"

if probability >= 0.70:
    probability_result = "Review"
else:
    probability_result = "Hold"

print("Case 1")
print("Rule-Based Result:", rule_result)
print("Probability-Based Result:", probability_result)
print("")

score = 68
documents_complete = True
probability = 0.74

if documents_complete == True and score >= 70:
    rule_result = "Review"
else:
    rule_result = "Hold"

if probability >= 0.70:
    probability_result = "Review"
else:
    probability_result = "Hold"

print("Case 2")
print("Rule-Based Result:", rule_result)
print("Probability-Based Result:", probability_result)
print("")

score = 91
documents_complete = False
probability = 0.88

if documents_complete == True and score >= 70:
    rule_result = "Review"
else:
    rule_result = "Hold"

if probability >= 0.70:
    probability_result = "Review"
else:
    probability_result = "Hold"

print("Case 3")
print("Rule-Based Result:", rule_result)
print("Probability-Based Result:", probability_result)
print("")
score = 70
documents_complete = True
probability = 0.70

if documents_complete == True and score >= 70:
    rule_result = "Review"
else:
    rule_result = "Hold"

if probability >= 0.70:
    probability_result = "Review"
else:
    probability_result = "Hold"

print("Case 4")
print("Rule-Based Result:", rule_result)
print("Probability-Based Result:", probability_result)
print("")


print("PART 2: Scholarship Checking System")
print("-----------------------------------")
print("Rules: Score must be 70 or more.")
print("Attendance must be 75% or more.")
print("Documents and recommendation letter must be complete.")
print("")

score = 82
documents_complete = True
attendance = 80
recommendation_submitted = True

if score >= 70 and documents_complete == True and attendance >= 75 and recommendation_submitted == True:
    result = "Review"
else:
    result = "Hold"

print("Test 1: All conditions pass")
print("Expected: Review")
print("Actual:", result)
print("Reason: All requirements are complete.")
print("")
score = 82
documents_complete = True
attendance = 70
recommendation_submitted = True

if score >= 70 and documents_complete == True and attendance >= 75 and recommendation_submitted == True:
    result = "Review"
else:
    result = "Hold"

print("Test 2: Attendance condition fails")
print("Expected: Hold")
print("Actual:", result)
print("Reason: Attendance is below 75%.")
print("")
score = 82
documents_complete = True
attendance = 80
recommendation_submitted = False

if score >= 70 and documents_complete == True and attendance >= 75 and recommendation_submitted == True:
    result = "Review"
else:
    result = "Hold"

print("Test 3: Recommendation condition fails")
print("Expected: Hold")
print("Actual:", result)
print("Reason: Recommendation letter is not submitted.")
print("")

score = 70
documents_complete = True
attendance = 80
recommendation_submitted = True

if score >= 70 and documents_complete == True and attendance >= 75 and recommendation_submitted == True:
    result = "Review"
else:
    result = "Hold"

print("Test 4: Score threshold boundary")
print("Expected: Review")
print("Actual:", result)
print("Reason: Score is exactly 70, so it passes.")
print("")


print("All four scholarship tests are completed.")
print("Note: All thresholds are sample assumptions for this lab.")
