def extract_facts(text):
    text = text.lower()
    facts = []

    if "privacy" in text:
        facts.append("Privacy Violation")

    if "arbitrary" in text or "unreasonable" in text:
        facts.append("Arbitrary State Action")

    if "criminal" in text or "murder" in text:
        facts.append("Criminal Liability")

    return facts
