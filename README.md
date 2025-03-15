# AIR DRAW

### Air Drawing: Gesture-Based Drawing Application  
A real-time, hands-free drawing application that allows users to create digital art using hand gestures captured via webcam. Leverages computer vision and machine learning for intuitive interaction.

**Key Features:**  
🖌️ **Gesture Controls:**
- ✌️ *Drawing Mode:* Index finger gesture to draw in air
- ✊ *Eraser Mode:* Index + Middle fingers to erase
- 🤏 Pinch gesture for quick actions (color change/save)

**🎨 Creative Tools:**
- several colors palette with edge UI
- Adjustable brush/eraser sizes (`+`/`-` keys)
- Undo/redo functionality (`Z` key)
- Canvas clearing (`C` key)

**🚀 Technical Highlights:**
- Hybrid tracking system with motion prediction
- Edge-aware drawing continuation
- Adaptive good algorithms
- Kalman filtering for stable tracking
- Real-time FPS optimization (10 FPS)

**⚙️ Core Technologies:**
- MediaPipe Hands for landmark detection
- OpenCV for image processing/overlays
- NumPy for canvas operations
- Tkinter for file dialogs

**🌟 Unique Capabilities:**
- Lockable modes to prevent accidental switching
- Velocity-based stroke prediction
- Visual gesture progress indicators
- Clickable UI elements via hand tracking

**Usage Scenarios:**
- Digital presentations/whiteboarding
- Touch-free art creation
- Accessibility tool for motor-impaired users
- Interactive teaching aid
- Gesture-controlled UI prototype

**System Requirements:**
- Webcam (720p+ recommended)
- Python 3.7+ with listed dependencies
- Moderate CPU/GPU resources

**Control Reference:**  
`Q`-Quit | `S`-Save | `M`-Toggle Mode | `L`-Lock Mode  
*Color selection via right-side palette interaction*

This project demonstrates practical implementation of real-time computer vision techniques for natural user interaction, suitable as a foundation for AR/VR input systems or interactive art installations.
