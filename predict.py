import joblib
from sentence_transformers import SentenceTransformer
import language_tool_python
import re
import sys


def clean_text(text):
    return re.sub(r'\s+', ' ', text.strip())

def get_error_counts(text):
    tool = language_tool_python.LanguageTool('en-US')
    matches = tool.check(text)
    
    categories = {
        'Spelling': 0,
        'White Space': 0,
        'Style': 0,
        'Grammar': 0,
        'Typographical': 0
    }

    for match in matches:
        rule = match.ruleIssueType.lower()
        if rule == 'misspelling':
            categories['Spelling'] += 1
        elif rule == 'whitespace':
            categories['White Space'] += 1
        elif rule == 'typographical':
            categories['Typographical'] += 1
        elif rule == 'grammar':
            categories['Grammar'] += 1
        elif rule == 'style':
            categories['Style'] += 1
        else:
            categories['Grammar'] += 1  # default fallback

    return categories

def calculate_score(errors):
    score = 100
    score -= errors['Spelling'] * 1.0
    score -= errors['White Space'] * 0.5
    score -= errors['Style'] * 0.5
    score -= errors['Grammar'] * 1.5
    score -= errors['Typographical'] * 0.5
    return max(0, round(score))

def generate_feedback(effectiveness, errors):
    feedback = []

    # Praise
    if effectiveness in ["Effective", "Adequate"]:
        feedback.append("Your essay has a clear structure and addresses the topic well.")
    if errors['Style'] == 0:
        feedback.append("Your tone and style are appropriate for formal writing.")
    if errors['Grammar'] <= 2:
        feedback.append("You used mostly correct grammar throughout your writing.")

    # Suggestions
    if errors['Spelling'] > 0:
        feedback.append(f"There are {errors['Spelling']} spelling errors. Consider proofreading.")
    if errors['Grammar'] > 2:
        feedback.append(f"Grammar needs some work—there are {errors['Grammar']} issues.")
    if errors['White Space'] > 0:
        feedback.append("Fix white space issues for better readability.")
    if errors['Typographical'] > 0:
        feedback.append("Check for typographical errors like repeated letters or characters.")

    return "\n".join(feedback)

def evaluate_essay(essay_text):
    essay_text = clean_text(essay_text)

    # Load components
    model = joblib.load("effectiveness_classifier_svm.pk2")
    encoder = joblib.load("label_encoder.pk2")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed and predict
    emb = embedder.encode([essay_text])
    pred = model.predict(emb)
    effectiveness = encoder.inverse_transform(pred)[0]

    # Error analysis
    errors = get_error_counts(essay_text)
    score = calculate_score(errors)
    feedback = generate_feedback(effectiveness, errors)

    # Output
    print("\n📝 Essay Evaluation")
    print(f"Effectiveness: {effectiveness}")
    print("\nError Summary:")
    for k, v in errors.items():
        print(f"{k}: {v}")
    print(f"\n✅ Overall Score: {score}/100")
    print("\n💬 Feedback:")
    print(feedback)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        essay_input = " ".join(sys.argv[1:])  # Combine all command-line args
    else:
        essay_input = input("Enter your essay text: ")

    evaluate_essay(essay_input)
