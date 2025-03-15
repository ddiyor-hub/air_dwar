import cv2
import numpy as np
import mediapipe as mp
import time
import os
import tkinter as tk
from collections import deque
import math
from tkinter import filedialog

# Configuration parameters
CONFIG = {
    'smooth_factor': 0.2,     # Уменьшен фактор сглаживания для меньшей задержки
    'max_history': 6,         # Уменьшена длина истории для более быстрого отклика
    'min_detection_confidence': 0.6,
    'min_tracking_confidence': 0.6,
    'cooldown_time': 0.8,
    'notification_duration': 1.5,
    'palette_width': 40,
    'max_thickness': 30,
    'colors': [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Purple
        (255, 255, 0),  # Cyan
        (255, 255, 255), # White
        (128, 0, 0),    # Dark Blue
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Dark Red
        (153, 51, 255), # Violet
        (255, 153, 51), # Orange
        (0, 255, 255),  # Aqua
        (0, 0, 0)       # Black
    ],
    'gesture_thresholds': {
        'erase_distance': 30,
        'save_distance': 0.05,
        'mode_switch_duration': 1.0,  # Время для удержания жеста перед переключением режима
        'color_change_min_duration': 0.6  # Минимальное время для смены цвета жестом
    },
    'prediction': {
        'enabled': True,         # Включить предсказание движения
        'look_ahead_factor': 0.2, # Насколько сильно предсказывать вперёд (0-1)
        'max_predict_distance': 20 # Максимальное расстояние предсказания
    },
    'edge_drawing': {
        'enabled': True,         # Автоматически включено рисование по краям
        'border_margin': 50,     # Расстояние от края считается границей (в пикселях)
        'extrapolation_factor': 1.3, # Насколько продлевать вектор движения за край
        'max_continue_time': 0.5  # Максимальное время продолжения рисования без обнаружения пальца
    },
    'ui': {
        'color_selection_cooldown': 0.5,  # Задержка между выборами цвета
        'palette_click_area_extension': 5  # Расширение зоны нажатия палитры
    }
}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class AirCanvas:
    def __init__(self, screen_width, screen_height):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=CONFIG['min_detection_confidence'],
            min_tracking_confidence=CONFIG['min_tracking_confidence']
        )
        
        # Application state
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        self.drawing_color = CONFIG['colors'][0]
        self.current_color_index = 0
        self.drawing_thickness = 8
        self.eraser_thickness = 50  # Much larger eraser
        self.mode = 'draw'
        self.mode_locked = False    # Режим заблокирован? (предотвращает случайное переключение)
        self.notification = {'text': '', 'time': 0}
        self.history = []
        self.max_undo = CONFIG['max_history']
        self.last_actions = {
            'color_change': 0,
            'clear': 0,
            'save': 0,
            'undo': 0,
            'mode_change': 0,
            'color_palette_click': 0
        }
        self.color_names = ["Red", "Green", "Blue", "Yellow", "Purple", "Cyan", "White", 
                            "Dark Blue", "Dark Green", "Dark Red", "Violet", "Orange", "Aqua", "Black"]
        
        # Переменные для отслеживания движения
        self.prev_draw_point = None
        self.prev_erase_point = None
        self.draw_points_history = deque(maxlen=CONFIG['max_history'])  # Уменьшено для меньшей задержки
        self.erase_points_history = deque(maxlen=CONFIG['max_history'] - 2)
        
        # Переменные для предсказания движения
        self.velocity = np.array([0, 0])  # Скорость движения (dx, dy)
        self.last_position = None
        self.last_position_time = 0
        self.predicted_point = None
        
        # Для рисования по краям экрана
        self.edge_drawing_mode = False
        self.last_finger_vector = np.array([0, 0])  # Вектор направления пальца
        self.last_detect_time = 0
        self.last_hand_landmarks = None
        self.is_near_edge = False
        self.edge_extrapolated_point = None
        
        # Плавное переключение режимов
        self.gesture_start_times = {
            'drawing': 0,
            'erasing': 0
        }
        self.current_gesture = None
        
        # Для цветовой палитры
        self.color_selection_active = False
        self.palette_areas = []  # Области для каждого цвета
        
        # Show initial notification
        self.show_notification("Air Canvas Ready")
    
    def save_state(self):
        """Save current canvas state to history stack"""
        if len(self.history) >= self.max_undo:
            self.history.pop(0)
        self.history.append(self.canvas.copy())

    def undo(self):
        """Restore previous canvas state"""
        if len(self.history) > 0:
            self.canvas = self.history.pop()
            self.show_notification("Undo")
            return True
        self.show_notification("Nothing to undo")
        return False

    def show_notification(self, text):
        """Display notification text on screen"""
        self.notification = {'text': text, 'time': time.time()}

    def update_velocity(self, current_pos):
        """Обновляет скорость движения на основе текущей и предыдущей позиции"""
        if self.last_position is None or current_pos is None:
            self.velocity = np.array([0, 0])
            self.last_position = current_pos
            self.last_position_time = time.time()
            return
        
        current_time = time.time()
        dt = current_time - self.last_position_time
        
        # Защита от деления на ноль
        if dt < 0.001:
            return
            
        dx = current_pos[0] - self.last_position[0]
        dy = current_pos[1] - self.last_position[1]
        
        # Фильтрация скорости с сохранением части предыдущего значения
        raw_velocity = np.array([dx / dt, dy / dt])
        smooth_factor = 0.7  # 70% новое значение, 30% старое
        self.velocity = smooth_factor * raw_velocity + (1 - smooth_factor) * self.velocity
        
        self.last_position = current_pos
        self.last_position_time = current_time

    def predict_next_position(self, current_pos):
        """Предсказывает следующую позицию на основе скорости движения"""
        if current_pos is None or np.linalg.norm(self.velocity) < 1:
            return current_pos
            
        # Предсказываем будущую позицию
        prediction_time = CONFIG['prediction']['look_ahead_factor']
        dx = int(self.velocity[0] * prediction_time)
        dy = int(self.velocity[1] * prediction_time)
        
        # Ограничиваем максимальное расстояние предсказания
        dist = math.sqrt(dx**2 + dy**2)
        if dist > CONFIG['prediction']['max_predict_distance']:
            scale = CONFIG['prediction']['max_predict_distance'] / dist
            dx = int(dx * scale)
            dy = int(dy * scale)
        
        predicted_x = min(max(0, current_pos[0] + dx), self.screen_width - 1)
        predicted_y = min(max(0, current_pos[1] + dy), self.screen_height - 1)
        
        return (predicted_x, predicted_y)

    def check_near_edge(self, point):
        """Проверяет, находится ли точка близко к краю экрана"""
        margin = CONFIG['edge_drawing']['border_margin']
        
        # Проверяем, близко ли к любому из четырех краев
        near_left = point[0] < margin
        near_right = point[0] > self.screen_width - margin
        near_top = point[1] < margin
        near_bottom = point[1] > self.screen_height - margin
        
        return near_left or near_right or near_top or near_bottom

    def extrapolate_to_edge(self, point, velocity_vector):
        """Экстраполирует точку до края экрана в направлении движения"""
        if point is None or np.linalg.norm(velocity_vector) < 0.1:
            return point
        
        # Нормализуем вектор и умножаем на коэффициент экстраполяции
        factor = CONFIG['edge_drawing']['extrapolation_factor']
        direction = velocity_vector / np.linalg.norm(velocity_vector)
        
        # Получаем экстраполированную точку
        ex_dx = int(direction[0] * factor * CONFIG['edge_drawing']['border_margin'])
        ex_dy = int(direction[1] * factor * CONFIG['edge_drawing']['border_margin'])
        
        # Ограничиваем точку пределами экрана
        ex_x = min(max(0, point[0] + ex_dx), self.screen_width - 1)
        ex_y = min(max(0, point[1] + ex_dy), self.screen_height - 1)
        
        return (ex_x, ex_y)

    def calculate_finger_vector(self, hand_landmarks):
        """Вычисляет вектор направления от ладони к кончику указательного пальца"""
        if not hand_landmarks:
            return np.array([0, 0])
            
        landmarks = hand_landmarks.landmark
        
        # Получаем положение ключевых точек
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        
        # Преобразуем в координаты экрана
        tip_x, tip_y = int(index_tip.x * self.screen_width), int(index_tip.y * self.screen_height)
        mcp_x, mcp_y = int(index_mcp.x * self.screen_width), int(index_mcp.y * self.screen_height)
        
        # Вычисляем вектор направления
        vector = np.array([tip_x - mcp_x, tip_y - mcp_y])
        
        # Нормализуем вектор
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector

    def detect_gestures(self, hand_landmarks, width, height):
        """Detect various hand gestures with improved edge support"""
        # Создаем базовый словарь с дефолтными значениями для всех ключей,
        # чтобы избежать KeyError
        result = {
            'drawing': False,
            'erasing': False,
            'drawing_long_enough': False,
            'erasing_long_enough': False,
            'change_color': False,
            'save': False,
            'position': None,
            'raw_position': None,
            'edge_mode': False
        }
        
        # Обработка случая, когда рука не обнаружена
        if not hand_landmarks:
            # Если мы в режиме рисования по краям, продолжаем использовать экстраполированные точки
            current_time = time.time()
            if (CONFIG['edge_drawing']['enabled'] and
                self.edge_drawing_mode and
                current_time - self.last_detect_time < CONFIG['edge_drawing']['max_continue_time']):
                # Продолжаем использовать последнее направление движения
                last_point = self.last_position if self.last_position else (width // 2, height // 2)
                edge_point = self.extrapolate_to_edge(last_point, self.velocity)
                self.edge_extrapolated_point = edge_point
                
                result.update({
                    'position': edge_point,
                    'drawing': True,  # Продолжаем рисовать
                    'edge_mode': True
                })
                return result
            else:
                # Обычный случай - рука не обнаружена
                self.current_gesture = None
                self.edge_drawing_mode = False
                return result
            
        # Обновляем время последнего обнаружения руки
        self.last_detect_time = time.time()
        self.last_hand_landmarks = hand_landmarks
            
        landmarks = hand_landmarks.landmark
        
        # Get key finger positions
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = landmarks[mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = landmarks[mp_hands.HandLandmark.PINKY_PIP]
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
        
        # Convert to screen coordinates - точно используем кончик пальца
        index_pos = (int(index_tip.x * width), int(index_tip.y * height))
        
        # Проверяем, находится ли палец близко к краю экрана
        self.is_near_edge = self.check_near_edge(index_pos)
        
        # Получаем вектор направления пальца для экстраполяции
        finger_vector = self.calculate_finger_vector(hand_landmarks)
        if np.linalg.norm(finger_vector) > 0:
            self.last_finger_vector = finger_vector
        
        # Экстраполируем точку до края, если палец близко к краю
        if self.is_near_edge and CONFIG['edge_drawing']['enabled']:
            self.edge_drawing_mode = True
            edge_point = self.extrapolate_to_edge(index_pos, self.last_finger_vector)
            self.edge_extrapolated_point = edge_point
        else:
            self.edge_drawing_mode = False
            self.edge_extrapolated_point = None
            edge_point = index_pos
        
        # Обновляем скорость движения
        self.update_velocity(index_pos)
        
        # Предсказываем следующую позицию с учетом края экрана
        if self.edge_drawing_mode:
            predicted_pos = edge_point
        else:
            predicted_pos = self.predict_next_position(index_pos) if CONFIG['prediction']['enabled'] else index_pos
        
        self.predicted_point = predicted_pos
        
        # Check if fingers are up (extended) or down (bent)
        index_up = index_tip.y < index_pip.y - 0.02
        middle_up = middle_tip.y < middle_pip.y - 0.02
        ring_up = ring_tip.y < ring_pip.y - 0.02
        pinky_up = pinky_tip.y < pinky_pip.y - 0.02
        thumb_up = thumb_tip.y < thumb_ip.y - 0.02
        
        # Calculate distance between index and middle finger tips
        index_middle_distance = np.sqrt(
            (index_tip.x - middle_tip.x)**2 + 
            (index_tip.y - middle_tip.y)**2
        )
        
        # Улучшенное обнаружение жеста стирания - два варианта
        tip_distance_erasing = (index_up and middle_up and 
                              index_middle_distance < 0.05 and
                              abs(index_tip.y - middle_tip.y) < 0.03)  # Пальцы должны быть примерно на одном уровне
                              
        pose_erasing = (index_up and middle_up and 
                       not ring_up and not pinky_up)
                       
        erasing_gesture = tip_distance_erasing or pose_erasing
        
        # Жест рисования: только указательный палец вверх
        drawing_gesture = (index_up and not middle_up and 
                          not ring_up and not pinky_up)
        
        # Жест сохранения (щипок) - больше не используется для сохранения
        pinch_distance = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )
        
        # Определяем текущий жест для отслеживания его продолжительности
        if drawing_gesture and not erasing_gesture:
            current_gesture = 'drawing'
        elif erasing_gesture and not drawing_gesture:
            current_gesture = 'erasing'
        else:
            current_gesture = None
            
        # Если жест изменился, обновляем время его начала
        if self.current_gesture != current_gesture:
            if current_gesture is not None:
                self.gesture_start_times[current_gesture] = time.time()
            self.current_gesture = current_gesture
        
        # Проверяем продолжительность жеста для переключения режима
        current_time = time.time()
        drawing_duration = current_time - self.gesture_start_times.get('drawing', 0)
        erasing_duration = current_time - self.gesture_start_times.get('erasing', 0)
        
        # Жест должен удерживаться нужное время для переключения режима
        drawing_long_enough = drawing_duration > CONFIG['gesture_thresholds']['mode_switch_duration']
        erasing_long_enough = erasing_duration > CONFIG['gesture_thresholds']['mode_switch_duration']
        
        result.update({
            'position': predicted_pos,  # Используем предсказанную или экстраполированную позицию
            'raw_position': index_pos,   # Сохраняем реальную позицию тоже
            'drawing': drawing_gesture,
            'drawing_long_enough': drawing_gesture and drawing_long_enough,
            'erasing': erasing_gesture,
            'erasing_long_enough': erasing_gesture and erasing_long_enough,
            'edge_mode': self.edge_drawing_mode
        })
        return result

    def handle_gestures(self, gestures, current_time):
        """Process detected gestures and update application state"""
        if not gestures or gestures.get('position') is None:
            return

        # Переключение режимов только после удержания жеста и только если режим не заблокирован
        if not self.mode_locked:  # Проверяем, не заблокирован ли режим
            if gestures.get('erasing_long_enough', False) and self.mode != 'erase' and current_time - self.last_actions['mode_change'] > CONFIG['cooldown_time']:
                self.mode = 'erase'
                self.show_notification("Eraser Mode")
                self.last_actions['mode_change'] = current_time
                # Reset drawing tracking
                self.prev_draw_point = None
                
            elif gestures.get('drawing_long_enough', False) and self.mode != 'draw' and current_time - self.last_actions['mode_change'] > CONFIG['cooldown_time']:
                self.mode = 'draw'
                self.show_notification("Drawing Mode")
                self.last_actions['mode_change'] = current_time
                # Reset eraser tracking
                self.prev_erase_point = None
        
        # Жест смены цвета удален - теперь цвет меняется только при наведении на палитру
        
        # Жест сохранения больше не используется - сохранение только по клавише S

    def smooth_point(self, new_point, history, max_points=5):
        """Apply minimal smoothing to reduce jitter but maintain responsiveness"""
        # Add new point to history
        history.append(new_point)
        
        # If history is empty or very short, just return the new point
        if len(history) <= 1:
            return new_point
        
        # Используем минимальное сглаживание для быстрого отклика
        weights = np.linspace(0.6, 1.0, len(history))
        weights = weights / np.sum(weights)
        
        x = int(np.sum([p[0] * w for p, w in zip(history, weights)]))
        y = int(np.sum([p[1] * w for p, w in zip(history, weights)]))
        
        return (x, y)
    
    def interpolate_points(self, p1, p2):
        """Interpolate points between p1 and p2 for smooth lines during fast movements"""
        if p1 is None or p2 is None:
            return []
            
        # Calculate distance between points
        dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # If distance is small, no need to interpolate
        if dist < 20:
            return []
            
        # Determine number of points to interpolate
        num_points = max(1, min(int(dist / 15), 10))
        
        # Create interpolated points
        interpolated = []
        for i in range(1, num_points):
            t = i / num_points
            x = int(p1[0] * (1 - t) + p2[0] * t)
            y = int(p1[1] * (1 - t) + p2[1] * t)
            interpolated.append((x, y))
            
        return interpolated
    
    def update_canvas(self, gestures):
        """Update canvas based on current gestures with edge support"""
        if not gestures or gestures.get('position') is None:
            # Reset tracking if no valid gestures
            return
        
        # Get current position (предсказанная или экстраполированная)
        current_position = gestures['position']
        raw_position = gestures.get('raw_position')
        
        # Handle drawing with minimal smoothing
        if self.mode == 'draw' and (gestures.get('drawing', False) or gestures.get('edge_mode', False)):
            # Apply minimal smoothing for faster response
            smooth_position = self.smooth_point(current_position, self.draw_points_history, 5)
            
            if self.prev_draw_point is None:
                # First point in stroke
                self.prev_draw_point = smooth_position
                # Draw a small circle at the first point for better visual
                cv2.circle(self.canvas, smooth_position, self.drawing_thickness//2, self.drawing_color, -1)
                return
            
            # Check for fast movement that might need interpolation
            dist = np.sqrt((smooth_position[0] - self.prev_draw_point[0])**2 + 
                          (smooth_position[1] - self.prev_draw_point[1])**2)
            
            # Interpolate if distance is large (fast movement)
            if dist > 20:
                interpolated_points = self.interpolate_points(self.prev_draw_point, smooth_position)
                for point in interpolated_points:
                    cv2.line(
                        self.canvas, 
                        self.prev_draw_point, 
                        point, 
                        self.drawing_color, 
                        self.drawing_thickness
                    )
                    self.prev_draw_point = point
            
            # Draw line on canvas
            cv2.line(
                self.canvas, 
                self.prev_draw_point, 
                smooth_position, 
                self.drawing_color, 
                self.drawing_thickness
            )
            
            # Update previous point immediately
            self.prev_draw_point = smooth_position
        elif self.mode == 'draw':
            # Reset drawing if not actively drawing
            self.prev_draw_point = None
            self.draw_points_history.clear()
            
        # Handle erasing
        if self.mode == 'erase' and gestures.get('erasing', False):
            # Apply less smoothing for erasing (more direct control)
            smooth_position = self.smooth_point(current_position, self.erase_points_history, 3)
            
            if self.prev_erase_point is None:
                # First point in erase stroke
                self.prev_erase_point = smooth_position
                # Erase at current position
                cv2.circle(self.canvas, smooth_position, self.eraser_thickness, (0,0,0), -1)
                return
            
            # Check for fast movement that might need interpolation
            dist = np.sqrt((smooth_position[0] - self.prev_erase_point[0])**2 + 
                          (smooth_position[1] - self.prev_erase_point[1])**2)
            
            # Interpolate if distance is large (fast movement)
            if dist > 20:
                interpolated_points = self.interpolate_points(self.prev_erase_point, smooth_position)
                for point in interpolated_points:
                    cv2.line(
                        self.canvas, 
                        self.prev_erase_point, 
                        point, 
                        (0,0,0), 
                        self.eraser_thickness * 2
                    )
                    cv2.circle(self.canvas, point, self.eraser_thickness, (0,0,0), -1)
                    self.prev_erase_point = point
                
            # Erase with thick line to connect last and current point
            cv2.line(
                self.canvas, 
                self.prev_erase_point, 
                smooth_position, 
                (0,0,0),  # Always black for eraser 
                self.eraser_thickness * 2  # Double thickness for better coverage
            )
            
            # Add circle at current position for better coverage
            cv2.circle(self.canvas, smooth_position, self.eraser_thickness, (0,0,0), -1)
            
            # Update previous point
            self.prev_erase_point = smooth_position
        elif self.mode == 'erase':
            # Reset erasing if not actively erasing
            self.prev_erase_point = None
            self.erase_points_history.clear()

    def save_canvas(self):
        """Save only the drawing canvas as PNG image file when requested by user"""
        # Create a Tkinter root window
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        # Ask the user where to save the file
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Save Drawing As"
        )
        
        # If the user cancels the save dialog
        if not file_path:
            return "Save cancelled"
        
        # Save only the drawing canvas (without camera feed or UI)
        drawing_only = self.canvas.copy()
        cv2.imwrite(file_path, drawing_only)
        
        # Clean up Tkinter
        root.destroy()
        
        return file_path
    
    def get_palette_areas(self):
        """Получает области для каждого цвета в палитре"""
        colors_count = len(CONFIG['colors'])
        palette_columns = 2 if colors_count > 8 else 1
        colors_per_column = (colors_count + palette_columns - 1) // palette_columns
        
        palette_areas = []
        for i in range(colors_count):
            column = i // colors_per_column
            row = i % colors_per_column
            
            # Координаты цветового квадрата с увеличенной областью нажатия
            x_start = self.screen_width - (CONFIG['palette_width'] + 10) * (palette_columns - column)
            y_start = 50 + row * (CONFIG['palette_width'] + 5)
            width = CONFIG['palette_width']
            height = CONFIG['palette_width']
            
            # Добавляем небольшое расширение зоны нажатия
            extension = CONFIG['ui']['palette_click_area_extension']
            palette_areas.append((
                x_start - extension, 
                y_start - extension, 
                width + 2*extension, 
                height + 2*extension,
                i  # Индекс цвета
            ))
            
        return palette_areas

    def draw_ui(self, frame):
        """Draw user interface elements on frame"""
        # Color palette - organize in two columns if many colors
        colors_count = len(CONFIG['colors'])
        palette_columns = 2 if colors_count > 8 else 1
        colors_per_column = (colors_count + palette_columns - 1) // palette_columns
        
        # Сохраняем области цветовой палитры для определения нажатий
        self.palette_areas = []
        
        for i, color in enumerate(CONFIG['colors']):
            column = i // colors_per_column
            row = i % colors_per_column
            
            # Calculate position based on column and row
            x_start = frame.shape[1] - (CONFIG['palette_width'] + 10) * (palette_columns - column)
            y_start = 50 + row * (CONFIG['palette_width'] + 5)
            x_end = x_start + CONFIG['palette_width']
            y_end = y_start + CONFIG['palette_width']
            
            # Сохраняем область для обнаружения нажатий
            self.palette_areas.append((x_start, y_start, CONFIG['palette_width'], CONFIG['palette_width'], i))
            
            # Конвертируем цвета для корректного отображения в OpenCV (BGR -> RGB)
            display_color = (color[2], color[1], color[0])
            
            # Draw color square
            cv2.rectangle(frame, 
                (x_start, y_start),
                (x_end, y_end), display_color, -1)
            
            # Highlight selected color
            if i == self.current_color_index:
                cv2.rectangle(frame, 
                    (x_start - 5, y_start - 5),
                    (x_end + 5, y_end + 5), (255, 255, 255), 2)
                
                # Дополнительная внутренняя рамка для контраста с белым
                cv2.rectangle(frame, 
                    (x_start - 3, y_start - 3),
                    (x_end + 3, y_end + 3), (0, 0, 0), 1)

        # Tool indicator (at bottom left)
        tool_bg_color = (40, 40, 40)
        tool_text_color = (255, 255, 255)
        if self.mode == 'draw':
            # Конвертируем цвет для отображения
            tool_icon_color = (self.drawing_color[2], self.drawing_color[1], self.drawing_color[0])
            tool_text = f"DRAW ({self.drawing_thickness}px)"
        else:
            tool_icon_color = (200, 200, 200)
            tool_text = f"ERASE ({self.eraser_thickness}px)"
            
        # Tool background
        cv2.rectangle(frame, (10, frame.shape[0] - 60), (200, frame.shape[0] - 10), tool_bg_color, -1)
        # Tool icon
        cv2.circle(frame, (30, frame.shape[0] - 35), 15, tool_icon_color, -1)
        # Tool text
        cv2.putText(frame, tool_text, (55, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, tool_text_color, 2)
        
        # Кнопки управления режимами
        # Кнопка M (Mode) - переключение режима
        mode_button_color = (0, 200, 0) if self.mode == 'draw' else (0, 0, 200)
        cv2.rectangle(frame, 
            (frame.shape[1] - 150, frame.shape[0] - 60),
            (frame.shape[1] - 110, frame.shape[0] - 20), 
            mode_button_color, -1)
        cv2.putText(frame, "M", 
            (frame.shape[1] - 138, frame.shape[0] - 35), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        # Подпись для кнопки M
        cv2.putText(frame, "Mode", 
            (frame.shape[1] - 150, frame.shape[0] - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        # Кнопка L (Lock) - блокировка режима
        lock_button_color = (0, 0, 200) if self.mode_locked else (150, 150, 150)
        cv2.rectangle(frame, 
            (frame.shape[1] - 90, frame.shape[0] - 60),
            (frame.shape[1] - 50, frame.shape[0] - 20), 
            lock_button_color, -1)
        cv2.putText(frame, "L", 
            (frame.shape[1] - 77, frame.shape[0] - 35), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
        # Подпись для кнопки L
        cv2.putText(frame, "Lock", 
            (frame.shape[1] - 90, frame.shape[0] - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Отображение текущего жеста и режима (индикатор прогресса)
        if self.current_gesture == 'drawing':
            gesture_time = time.time() - self.gesture_start_times.get('drawing', 0)
            if gesture_time < CONFIG['gesture_thresholds']['mode_switch_duration'] and self.mode != 'draw' and not self.mode_locked:
                # Отображаем прогресс переключения режима
                progress = min(1.0, gesture_time / CONFIG['gesture_thresholds']['mode_switch_duration'])
                bar_width = int(100 * progress)
                cv2.rectangle(frame, (10, 70), (110, 90), (40, 40, 40), -1)
                cv2.rectangle(frame, (10, 70), (10 + bar_width, 90), (0, 255, 0), -1)
                cv2.putText(frame, "To DRAW", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        elif self.current_gesture == 'erasing':
            gesture_time = time.time() - self.gesture_start_times.get('erasing', 0)
            if gesture_time < CONFIG['gesture_thresholds']['mode_switch_duration'] and self.mode != 'erase' and not self.mode_locked:
                # Отображаем прогресс переключения режима
                progress = min(1.0, gesture_time / CONFIG['gesture_thresholds']['mode_switch_duration'])
                bar_width = int(100 * progress)
                cv2.rectangle(frame, (10, 70), (110, 90), (40, 40, 40), -1)
                cv2.rectangle(frame, (10, 70), (10 + bar_width, 90), (200, 0, 0), -1)
                cv2.putText(frame, "To ERASE", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        

        
        # Статус блокировки режима
        if self.mode_locked:
            cv2.putText(frame, "MODE LOCKED", 
                      (frame.shape[1] - 150, 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Instructions (top of screen)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (40, 40, 40), -1)
        cv2.putText(frame, "Index finger: DRAW | Index+Middle: ERASE | S: Save | C: Clear | Q: Quit | Z: Undo", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Notification display
        if time.time() - self.notification['time'] < CONFIG['notification_duration']:
            text_size = cv2.getTextSize(self.notification['text'], 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            # Semi-transparent overlay
            overlay = frame.copy()
            # Background
            cv2.rectangle(overlay, 
                (frame.shape[1]//2 - text_size[0]//2 - 25, frame.shape[0]//2 - 25),
                (frame.shape[1]//2 + text_size[0]//2 + 25, frame.shape[0]//2 + 25),
                (0, 0, 0), -1)
            # Apply transparency
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            # Border
            cv2.rectangle(frame, 
                (frame.shape[1]//2 - text_size[0]//2 - 25, frame.shape[0]//2 - 25),
                (frame.shape[1]//2 + text_size[0]//2 + 25, frame.shape[0]//2 + 25),
                (255, 255, 255), 1)
            # Text
            cv2.putText(frame, self.notification['text'], 
                       (frame.shape[1]//2 - text_size[0]//2, frame.shape[0]//2 + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


def check_button_click(point, rect):
    """Проверяет, нажата ли кнопка"""
    x, y = point
    rect_x, rect_y, rect_w, rect_h = rect
    return rect_x <= x <= rect_x + rect_w and rect_y <= y <= rect_y + rect_h


def main():
    # Get screen resolution
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    # Setup fullscreen window
    cv2.namedWindow('Air Drawing Pro', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Air Drawing Pro', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access camera")
        return
    
    # Улучшенная настройка камеры для минимальной задержки
    try:
        cap.set(cv2.CAP_PROP_FPS, 60)  # Максимальный FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Минимальный буфер для меньшей задержки
    except:
        print("Warning: Could not set camera parameters")

    # Initialize Air Canvas with screen dimensions
    air_canvas = AirCanvas(screen_width, screen_height)
    air_canvas.save_state()  # Save initial empty state
    
    # Main loop variables
    frame_count = 0
    need_to_save_state = False
    fps_history = deque(maxlen=30)
    last_frame_time = time.time()
    fps = 0
    
    # Определение областей для кнопок
    mode_button_rect = (screen_width - 150, screen_height - 60, 40, 40)  # x, y, width, height
    lock_button_rect = (screen_width - 90, screen_height - 60, 40, 40)

    print("Air Drawing Pro started. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break

        # Вычисление FPS
        current_time = time.time()
        delta_time = current_time - last_frame_time
        last_frame_time = current_time
        
        if delta_time > 0:
            current_fps = 1.0 / delta_time
            fps_history.append(current_fps)
            fps = sum(fps_history) / len(fps_history)

        # Process frame
        frame = cv2.flip(frame, 1)  # Mirror image for more intuitive drawing
        frame = cv2.resize(frame, (screen_width, screen_height))
        
        # Обрабатываем каждый кадр для минимальной задержки
        frame_count += 1
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process with MediaPipe
        results = air_canvas.hands.process(rgb_frame)
        current_time = time.time()
        
        # Detect and handle gestures
        hand_landmarks = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]  # Use first hand only
            
            # Draw hand landmarks on frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        
        # Process gestures
        gestures = air_canvas.detect_gestures(hand_landmarks, screen_width, screen_height)
        
        # Проверяем нажатие кнопок и палитры
        if hand_landmarks:
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            point = (int(index_tip.x * screen_width), int(index_tip.y * screen_height))
            
            # Проверяем кнопку режима (M)
            if check_button_click(point, mode_button_rect) and current_time - air_canvas.last_actions['mode_change'] > CONFIG['cooldown_time']:
                air_canvas.mode = 'erase' if air_canvas.mode == 'draw' else 'draw'
                air_canvas.show_notification(f"Mode: {air_canvas.mode.upper()}")
                air_canvas.last_actions['mode_change'] = current_time
                air_canvas.prev_draw_point = None
                air_canvas.prev_erase_point = None
            
            # Проверяем кнопку блокировки (L)
            elif check_button_click(point, lock_button_rect) and current_time - air_canvas.last_actions['mode_change'] > CONFIG['cooldown_time']:
                air_canvas.mode_locked = not air_canvas.mode_locked
                air_canvas.show_notification(f"Mode {'Locked' if air_canvas.mode_locked else 'Unlocked'}")
                air_canvas.last_actions['mode_change'] = current_time
                
            # Проверяем нажатие на цветовую палитру
            elif current_time - air_canvas.last_actions['color_palette_click'] > CONFIG['ui']['color_selection_cooldown']:
                for area in air_canvas.palette_areas:
                    x, y, w, h, color_index = area
                    if x <= point[0] <= x + w and y <= point[1] <= y + h:
                        # Выбор цвета из палитры
                        air_canvas.current_color_index = color_index
                        air_canvas.drawing_color = CONFIG['colors'][color_index]
                        air_canvas.show_notification(f"Color: {air_canvas.color_names[color_index]}")
                        air_canvas.last_actions['color_palette_click'] = current_time
                        break
        
        air_canvas.handle_gestures(gestures, current_time)
        
        # Determine if we should save state (if drawing or erasing occurred)
        prev_mode = air_canvas.mode
        prev_drawing = air_canvas.prev_draw_point is not None
        prev_erasing = air_canvas.prev_erase_point is not None
        
        # Update canvas based on gestures
        air_canvas.update_canvas(gestures)
        
        # Check if we should save state
        if prev_mode != air_canvas.mode:
            need_to_save_state = True
        elif (prev_drawing != (air_canvas.prev_draw_point is not None) or 
              prev_erasing != (air_canvas.prev_erase_point is not None)):
            need_to_save_state = True
        
        # Save state if needed (no auto-save)
        if need_to_save_state:
            air_canvas.save_state()
            need_to_save_state = False
        
        # Blend canvas with video frame (с корректным отображением цветов)
        canvas_display = air_canvas.canvas.copy()
        frame_with_canvas = cv2.addWeighted(frame, 0.7, canvas_display, 0.7, 0)
        
        # Draw UI elements
        air_canvas.draw_ui(frame_with_canvas)
        
        # Отображение FPS
        cv2.putText(frame_with_canvas, f"FPS: {int(fps)}", 
                   (10, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the result
        cv2.imshow('Air Drawing Pro', frame_with_canvas)

        # Handle keyboard input
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('z'):
            air_canvas.undo()
        elif key == ord('c'):
            air_canvas.save_state()
            air_canvas.canvas = np.zeros_like(air_canvas.canvas)
            air_canvas.show_notification("Canvas cleared")
        elif key in [ord('+'), ord('=')]:
            if air_canvas.mode == 'erase':
                air_canvas.eraser_thickness = min(air_canvas.eraser_thickness + 5, 100)
                air_canvas.show_notification(f"Eraser size: {air_canvas.eraser_thickness}")
            else:
                air_canvas.drawing_thickness = min(air_canvas.drawing_thickness + 1, CONFIG['max_thickness'])
                air_canvas.show_notification(f"Brush size: {air_canvas.drawing_thickness}")
        elif key == ord('-'):
            if air_canvas.mode == 'erase':
                air_canvas.eraser_thickness = max(air_canvas.eraser_thickness - 5, 10)
                air_canvas.show_notification(f"Eraser size: {air_canvas.eraser_thickness}")
            else:
                air_canvas.drawing_thickness = max(air_canvas.drawing_thickness - 1, 1)
                air_canvas.show_notification(f"Brush size: {air_canvas.drawing_thickness}")
        elif key == ord('s'):
            filename = air_canvas.save_canvas()
            air_canvas.show_notification(f"Drawing saved: {filename}")
        elif key == ord('m'):
            air_canvas.mode = 'erase' if air_canvas.mode == 'draw' else 'draw'
            air_canvas.show_notification(f"Mode: {air_canvas.mode.upper()}")
        elif key == ord('l'):
            # Переключение блокировки режима
            air_canvas.mode_locked = not air_canvas.mode_locked
            air_canvas.show_notification(f"Mode {'Locked' if air_canvas.mode_locked else 'Unlocked'}")

    # Clean up before exit
    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()