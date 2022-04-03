from Aurras import Aurras

aurras = Aurras()
aurras.load()

print("\nPositive Similarity, known intents")
aurras.get_intent("What do I have to do today?", "What reminders do I have?")

print("\nNegative Similarity, known intents")
aurras.get_intent("What do I have to do today?", "Play some Jazz")

print("\nPositive Similarity, unknown intents")
aurras.get_intent("Book me an uber to downtown", "Please setup an uber to get to school")

print("\nNegative Similarity, unknown intents")
aurras.get_intent("Open visual studio code", "Text Jason and tell him that I'll be a little late")