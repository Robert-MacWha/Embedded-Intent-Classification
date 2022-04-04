from Aurras import Aurras

aurras = Aurras()
aurras.load()

aurras.add_intent([
    "Start google chrome",
    "Load up flowkey",
    "Open visual studio code please"
], name="open_program")

print(aurras.get_intent("start celeste"))

aurras.save()