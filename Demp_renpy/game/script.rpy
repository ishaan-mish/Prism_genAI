# script.rpy

# Declare characters
define e = Character("Emily")

# Declare images
image bg room = "room.jpg"
image emily happy = "emily_happy.png"
image emily sad = "emily_sad.png"

# The game starts here
label start:

    scene bg room with fade

    show emily happy
    e "Hi there! Welcome to your first Ren'Py game."

    e "I’m Emily, your guide for today."

    "You see Emily standing in the room. She looks at you expectantly."

    menu:
        "How do you respond?"
        "Say hello":
            jump hello_path
        "Stay silent":
            jump silent_path

label hello_path:
    e "Oh! Hello to you too. I'm so glad you're friendly!"
    show emily happy
    "Emily smiles warmly."
    return

label silent_path:
    show emily sad
    e "Oh… I see. Maybe you don’t feel like talking."
    "Emily looks a little sad."
    return
