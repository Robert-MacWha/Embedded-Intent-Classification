from Aurras import Aurras

aurras = Aurras()
aurras.load()

print("Positive Similarity, known")
aurras.get_intent("What do I have to do today?", "What reminders do I have?")

print("vNegative Similarity, known")
aurras.get_intent("What do I have to do today?", "Play some Jazz")

print("\nPositive Similarity, unknown")
aurras.get_intent("Book me an uber to downtown", "Please setup an uber to get to school")

print("\nPositive Similarity, unknown")
aurras.get_intent("Start playing a short hike", "Open up cities skylines on steam")

print("\nNegative Similarity, unknown")
aurras.get_intent("Book me an uber to downtown", "Open up cities skylines on steam")