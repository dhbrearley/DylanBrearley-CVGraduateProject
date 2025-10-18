# 10/15/2025
- Left off trying to figure out what the worst acceptable darkness, brightness, blur range could be.
  
- Couldn't figure out how to run the PIV on a different video (update from 10/16: WSL required more RAM).

- Remember to source the virtual environment in root/bin/activate.
# 10/16/2025
- Succesfully created good and bad datasets but still have limited data to train on.

- Train the model using the training file under utils, it works best for up close videos of river.

- It will be best if I just use the model as a bad frame detector, then I need to take the average over a video and then define a threshold like >20% of frames bad then throw the video out.

- This combined with the rain drop detection for cascading machine learning models should be good progress.
# 10/17/2025
- The model added two new parameters for clipped white and black frames, while it did not significantly improve accuracy, it did improve accuracy of bad frame detection

- The model was tested on testVideo.py and succesfully generates a bad frame prediction ratio

- The next step is to start working on the raindrop detection algorithm so this will make the model a true cascading architecture
