# terrain_generator.py
import numpy as np

def create_terrain(x_size, y_size, terrain_type='flat'):
    x = np.arange(0, x_size)
    y = np.arange(0, y_size)
    xx, yy = np.meshgrid(x, y)
    
    if terrain_type == 'flat':
        z = np.zeros((y_size, x_size))
    elif terrain_type == 'puddle':
        center_x, center_y = x_size // 2, y_size // 2
        radius = min(x_size, y_size) / 4
        dist = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        z = -10 * np.exp(-(dist**2 / (2 * radius**2)))
    elif terrain_type == 'hill':
        center_x, center_y = x_size // 2, y_size // 2
        radius = min(x_size, y_size) / 3
        dist = np.sqrt((xx - center_x)**2 + (yy - center_y)**2)
        z = 30 * np.exp(-(dist**2 / (2 * radius**2)))
    else:
        z = np.zeros((y_size, x_size))
        
    return z