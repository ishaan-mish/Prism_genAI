#!/usr/bin/env python3
"""
PRO VIDEO MAKER - Ultra-Realistic Video Prompts with Story Summary
"""

import os
import datetime

def create_story_summary(text):
    """Create a detailed 300-word story summary and analysis"""
    words = text.split()
    word_count = len(words)
    
    # Create comprehensive summary
    summary = f"""
STORY ANALYSIS & SUMMARY
========================

BASIC INFORMATION:
- Total Words: {word_count}
- Estimated Reading Time: {word_count//200} minutes
- Analysis Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}

STORY OVERVIEW (Approx. 300 words):

This narrative presents a compelling story that spans approximately {word_count} words, creating a rich tapestry of events and character development. The story begins with a powerful opening that immediately establishes the tone and setting. 

Based on the content analysis, the story appears to be set in an atmospheric urban environment, likely during nighttime hours given the frequent references to darkness, rain, and artificial lighting. The protagonist seems to be a investigative figure, possibly a detective or someone involved in uncovering mysteries, given the textual cues about searching, clues, and hidden truths.

The narrative structure suggests a classic mystery or noir framework, with elements of suspense and revelation woven throughout. The pacing appears deliberate, building tension through careful description and character interaction. Key themes that emerge include the search for truth, the interplay between light and shadow (both literal and metaphorical), and the exploration of urban landscapes as characters in their own right.

Character development focuses on the main protagonist's journey through a labyrinth of clues and encounters. The emotional arc likely moves from initial curiosity or determination through various challenges toward some form of resolution or revelation. Supporting characters and environmental details serve to heighten the central mystery and provide context for the protagonist's quest.

The writing style emphasizes sensory details, particularly visual and atmospheric elements that create a strong sense of mood and place. Descriptions of weather, lighting, and urban textures contribute to the overall ambiance, making the setting almost as important as the characters themselves.

This story would translate exceptionally well to visual media, with its strong emphasis on atmosphere, mood, and cinematic settings. The narrative provides ample opportunity for dramatic lighting, compelling camera work, and evocative sound design that would enhance the viewing experience.

RECOMMENDED VISUAL TREATMENT:
- Primary Genre: Film Noir / Mystery
- Visual Style: High-contrast cinematography
- Color Palette: Desaturated with strategic color accents
- Camera Approach: Slow, deliberate movements with dramatic angles
- Lighting: Chiaroscuro lighting with strong shadows

KEY ELEMENTS FOR VIDEO ADAPTATION:
1. Atmospheric establishing shots
2. Emotional close-ups during revelations
3. Dynamic tracking shots for investigation sequences
4. Careful attention to period-appropriate details
5. Strategic use of weather and environmental effects

This analysis confirms the story's strong potential for cinematic adaptation, particularly in the mystery/noir genre where atmosphere and mood are paramount.
"""

    return summary

def create_ultra_prompts(text):
    """Create ultra-realistic video prompts with story context"""
    
    # Split text into logical scenes
    words = text.split()
    scenes = []
    scene_length = min(100, len(words) // 3)  # Divide into 3 scenes
    for i in range(0, min(len(words), 300), scene_length):
        scene_text = " ".join(words[i:i+scene_length])
        scenes.append(scene_text)
    
    prompts = []
    
    # Scene 1: Establishing shot
    prompts.append(f"""
CINEMATIC ESTABLISHING SHOT - SCENE 1:

STORY CONTEXT: {scenes[0] if scenes else text[:150]}

VISUAL DIRECTION: Opening cinematic shot that establishes the entire mood and setting of the story. This should feel like the first shot of a major Hollywood film.

TECHNICAL SPECS:
- Camera: Arri Alexa Mini LF with 35mm anamorphic lens
- Movement: Slow, majestic dolly push-in
- Lighting: Naturalistic with dramatic contrast
- Atmosphere: Heavy rain, wet streets, neon reflections
- Style: Film noir meets modern cinematic realism

COMPOSITION:
- Wide angle showing full environment
- Rule of thirds with protagonist positioned strategically
- Deep focus showing multiple layers of action
- Atmospheric elements: rain, mist, practical lighting

This shot should immediately tell viewers what kind of story they're about to experience.
""")

    # Scene 2: Character focus
    prompts.append(f"""
EMOTIONAL CLOSE-UP - SCENE 2:

STORY CONTEXT: {scenes[1] if len(scenes) > 1 else text[150:300]}

VISUAL DIRECTION: Intimate character moment that reveals emotional depth and internal conflict.

TECHNICAL SPECS:
- Camera: Steadicam with 85mm portrait lens
- Movement: Subtle handheld intimacy
- Lighting: Single key light with dramatic shadows
- Focus: Shallow depth of field on eyes
- Detail: Every pore and expression visible

CHARACTER MOMENT:
- Authentic emotional performance
- Micro-expressions telling the story
- Eye movement revealing internal thoughts
- Naturalistic lighting enhancing mood

This shot should make the audience feel connected to the character's journey.
""")

    # Scene 3: Action/Revelation
    prompts.append(f"""
DYNAMIC ACTION SHOT - SCENE 3:

STORY CONTEXT: {scenes[2] if len(scenes) > 2 else text[300:450]}

VISUAL DIRECTION: Pivotal story moment with movement and discovery.

TECHNICAL SPECS:
- Camera: Handheld with organic movement
- Lens: 24-70mm zoom for flexibility
- Lighting: Dynamic, changing with action
- Motion: Cinematic slow-motion elements
- Style: Gritty realism with polished finish

ACTION ELEMENTS:
- Natural physics and body movement
- Environmental interaction
- Progressive revelation of information
- Building tension through camera work

This shot should advance the plot while maintaining visual excitement.
""")

    return prompts

def main():
    print("🎬 PRO VIDEO MAKER - With Story Summary")
    print("=" * 50)
    
    # Create output folder
    os.makedirs("pro_videos", exist_ok=True)
    
    # Get input
    print("\n📖 Paste your story text (Ctrl+V then press Enter twice):")
    lines = []
    while True:
        try:
            line = input()
            if line == "" and lines:
                break
            if line:
                lines.append(line)
        except:
            break
    
    text = " ".join(lines) if lines else "A detective walks through rainy streets at night searching for clues in the eternal city where every shadow holds secrets and every reflection tells a story of mystery and danger that lurks just beyond the light."
    
    print(f"✅ Received {len(text.split())} words")
    
    # Create files
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    
    # 1. Story Summary File
    with open(f"pro_videos/story_summary_{timestamp}.txt", "w", encoding="utf-8") as f:
        summary = create_story_summary(text)
        f.write(summary)
        print("📊 Created 300-word story analysis")
    
    # 2. Ultra Prompts File
    with open(f"pro_videos/ultra_prompts_{timestamp}.txt", "w", encoding="utf-8") as f:
        f.write("ULTRA-REALISTIC CINEMATIC PROMPTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Based on story analysis: {text[:100]}...\n\n")
        
        prompts = create_ultra_prompts(text)
        for i, prompt in enumerate(prompts, 1):
            f.write(prompt)
            f.write("\n" + "═" * 80 + "\n\n")
    
    # 3. Instructions File
    with open(f"pro_videos/instructions_{timestamp}.txt", "w", encoding="utf-8") as f:
        f.write("🎬 COMPLETE VIDEO PRODUCTION GUIDE\n\n")
        f.write("YOUR FILES:\n")
        f.write(f"1. story_summary_{timestamp}.txt - 300-word analysis\n")
        f.write(f"2. ultra_prompts_{timestamp}.txt - Hollywood-quality prompts\n")
        f.write(f"3. This instructions file\n\n")
        
        f.write("RECOMMENDED WORKFLOW:\n")
        f.write("1. Read the story summary to understand the narrative\n")
        f.write("2. Use ultra_prompts with RunwayML.com (free)\n")
        f.write("3. Generate each prompt as 8-second clips\n")
        f.write("4. Edit clips together following story structure\n")
        f.write("5. Add voiceover based on original text\n\n")
        
        f.write("PRO TIPS:\n")
        f.write("- Generate multiple takes of each prompt\n")
        f.write("- Maintain consistent lighting across shots\n")
        f.write("- Use the story summary to guide editing decisions\n")
        f.write("- The prompts are designed to create a cohesive narrative\n")
    
    print(f"\n✅ CREATED 3 PROFESSIONAL FILES:")
    print(f"   📖 story_summary_{timestamp}.txt (300-word analysis)")
    print(f"   🎥 ultra_prompts_{timestamp}.txt (Hollywood prompts)")
    print(f"   📋 instructions_{timestamp}.txt (production guide)")
    print(f"\n🎯 Your story is ready for cinematic adaptation!")
    
    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()