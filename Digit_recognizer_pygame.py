from tkinter.font import Font
import pygame
import sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2
from tensorflow import keras
winx_size = 500
winy_size = 500
col_white = (255, 255, 255)
col_black = (0, 0, 0)
# Initialisinig pygame window
pygame.init()

# Defining display window
Display_surface = pygame.display.set_mode((winx_size, winy_size))

# Loading already trained model
digit_model = load_model('bestmodelachieved.h5')

# Defining tags for numbers to display
digit_tag = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
             5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

# Caption of window to appear
pygame.display.set_caption("Real time digit teller")

# Stores x and y coordinates of  region where you draw
array_x_coordinate = []
array_y_coordinate = []

# State of currently drawing, default=True
curr_drawing = False

# Infinite loop to keep running untill user stops
while(1):
    # Captures any input happening by mouse or keyboard
    for event_happening in pygame.event.get():
        if event_happening.type == QUIT:
            pygame.quit()
            sys.exit()

        # If mouse currently drawing,capture coordinates
        if event_happening.type == MOUSEMOTION and curr_drawing:
            x_coordinate, y_coordinate = event_happening.pos  # x and y coordinates
            # draws on board with white color
            pygame.draw.circle(Display_surface, col_white,
                               (x_coordinate, y_coordinate), 3, 0)

            # Pushes coordinates to array
            array_x_coordinate.append(x_coordinate)
            array_y_coordinate.append(y_coordinate)

        # Start capturing as soon as user starts drawing
        if event_happening.type == MOUSEBUTTONDOWN:
            curr_drawing = 1

        # Stop capturing and process till drawn image
        if event_happening.type == MOUSEBUTTONUP:
            curr_drawing = 0

            # Sorting x and y coordinates to capture pixels
            array_x_coordinate = sorted(array_x_coordinate)
            array_y_coordinate = sorted(array_y_coordinate)

            boundary_pixels = 2  # defines edge
            # stores x and y coordinates of 2 extreme locations to get pixel array
            rectangle_min_x_coordinate, rectangle_max_x_coordinate = max(
                array_x_coordinate[0] - boundary_pixels, 0), min(winx_size, array_x_coordinate[-1]+boundary_pixels)
            rectangle_min_y_coordinate, rectangle_max_y_coordinate = max(
                array_y_coordinate[0]-boundary_pixels, 0), min(winy_size, array_y_coordinate[-1]+boundary_pixels)

            # Initialising back to null
            array_x_coordinate = []
            array_y_coordinate = []

            # Getting pixels of the region described by rectangle
            image_array = np.array(pygame.PixelArray(Display_surface))[
                rectangle_min_x_coordinate:rectangle_max_x_coordinate, rectangle_min_y_coordinate:rectangle_max_y_coordinate].T.astype(np.float32)

            # converting to 28x28, no need to convert to gray image as already in gray
            resized_image = cv2.resize(
                image_array, (28, 28))

            # Converting to model readable format
            normalized_image = keras.utils.normalize(resized_image, axis=1)

            # Adding 1 more dimesnion for kernel operation
            normalized_image = np.array(
                normalized_image).reshape(-1, 28, 28, 1)

            # digit now contains info about the digit drawn
            digit = digit_tag[np.argmax(digit_model.predict(normalized_image))]

            # Defining font type
            FONT = pygame.font.Font("ShortBaby-Mg2w.ttf", 18)

            # Writing digit in font format that is with red color on white
            textarea = FONT.render(digit, True, (255, 0, 0), (255, 255, 255))

            # To display digit on small rectangle
            text_rectangle_object = textarea.get_rect()

            # Defining coordinates
            text_rectangle_object.left, text_rectangle_object.bottom = rectangle_min_x_coordinate, rectangle_max_y_coordinate

            # displaying rectangle with text on screen
            Display_surface.blit(textarea, text_rectangle_object)

        # If pressing q to clear screen
        if event_happening.type == KEYDOWN:
            if event_happening.unicode == "q":
                Display_surface.fill(col_black)

        # update display screen after every iteration
        pygame.display.update()
