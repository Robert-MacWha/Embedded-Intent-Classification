from Aurras import Aurras

aurras = Aurras()
aurras.load()

aurras.add_intent([
    "Start google chrome",
    "Load up flowkey",
    "Open visual studio code please"
], name="open_program")

aurras.add_intent([
    "Play Garden of Memories by Seycare",
    "Put on Imaginary expression by rush garcia",
    "Play some jazz music"  
], name="play_music")

aurras.add_intent([
    "What does my calendar look like for today",
    "What do I have to do today",
    "What's on my calendar"
], name="get_calendar")

aurras.save()
