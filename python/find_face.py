import cv2
#from PIL import Image
import dlib
from pathlib import Path
import math
from tqdm import tqdm
#from transformers.pipelines.base import Dataset

from cut_video import time_to_sec
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle
from transformers import AutoTokenizer

#from tokenizer import input_ids


def shape_to_list(shape):
	coords = []
	for i in range(0, 68):
		coords.append((shape.part(i).x, shape.part(i).y))
	return coords

def process_video(video_path):
	# Face detector and landmark detector
	face_detector = dlib.get_frontal_face_detector()
	landmark_detector = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Landmark detector path

	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

	while True:
		ret, frame = cap.read()
		if ret:
			face_rects = face_detector(frame, 1)  # Detect face
			if len(face_rects) == 0:
				out.write(frame)
				continue
			rect = face_rects[0]  # Proper number of face
			landmark = landmark_detector(frame, rect)  # Detect face landmarks

			landmark = shape_to_list(landmark)

			height = rect.bottom() - rect.top()
			width = rect.right() - rect.left()
			cv2.rectangle(frame, (rect.left(), rect.top()),(rect.left()+width, rect.top()+height),(0, 0, 255),thickness=2)

			LIP_MARGIN = 0.45  # Marginal rate for lip-only image.
			lip_landmark = landmark[48:68]  # Landmark corresponding to lip
			lip_x = sorted(lip_landmark, key=lambda pointx: pointx[0])  # Lip landmark sorted for determining lip region
			lip_y = sorted(lip_landmark, key=lambda pointy: pointy[1])
			x_add = int((-lip_x[0][0] + lip_x[-1][0]) * LIP_MARGIN)  # Determine Margins for lip-only image
			y_add = int((-lip_y[0][1] + lip_y[-1][1]) * LIP_MARGIN)
			crop_pos = (lip_x[0][0] - x_add, lip_x[-1][0] + x_add, lip_y[0][1] - y_add, lip_y[-1][1] + y_add)

			cv2.rectangle(frame, (lip_x[0][0] - x_add, lip_y[0][1] - y_add), ((lip_x[0][0] - x_add + (lip_x[-1][0] - lip_x[0][0] + 2 * x_add)),
						  (lip_y[0][1] - y_add + (lip_y[-1][1] - lip_y[0][1] + 2 * y_add))), (44, 160, 44), 2)
			out.write(frame)
	cap.release()
	out.release()

class LipReading2Preprocessor:
	def __init__(self, raw_data_path, processed_data_path, tokenizer_name="gpt2"):
		self.raw_data_path = Path(raw_data_path) # существующий
		self.processed_data_path = Path(processed_data_path) # обработанные файлы
		self.video_path = self.raw_data_path / "video"
		self.text_path = self.raw_data_path / "transcript"
		self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

		self.face_detector = dlib.get_frontal_face_detector()
		predictor_path = "shape_predictor_68_face_landmarks.dat"
		if not Path(predictor_path).exists():
			raise FileNotFoundError(f"Файл {predictor_path} не найден в директории проекта!")

		self.landmark_detector = dlib.shape_predictor(predictor_path)

	def extract_mouth_roi(self, frame):
		"""Выделение области рта из кадра"""
		# Конвертируем в grayscale для детектора
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Детекция лиц
		face_rects = self.face_detector(gray_frame, 1)

		# Проверяем, что найдено хотя бы одно лицо
		if len(face_rects) == 0:
			print("Лица не обнаружены")
			return None

		# Берем первое обнаруженное лицо
		rect = face_rects[0]

		# Детекция landmarks
		try:
			landmarks = self.landmark_detector(gray_frame, rect)
			landmarks_list = shape_to_list(landmarks)
		except Exception as e:
			print(f"Ошибка при детекции landmarks: {e}")
			return None

		# Целевые размеры и пропорции
		TARGET_WIDTH = 96
		TARGET_HEIGHT = 64
		TARGET_ASPECT_RATIO = TARGET_WIDTH / TARGET_HEIGHT  # 1.5
		LIP_MARGIN = 0.3  # Уменьшенный margin для более точного выделения

		# Landmarks для губ (48-67 - это контур губ)
		lip_landmarks = landmarks_list[48:68]

		# Проверяем, что landmarks для губ получены
		if len(lip_landmarks) == 0:
			print("Landmarks губ не обнаружены")
			return None

		# Получаем bounding box для губ
		lip_x_coords = [point[0] for point in lip_landmarks]
		lip_y_coords = [point[1] for point in lip_landmarks]

		x_min = min(lip_x_coords)
		x_max = max(lip_x_coords)
		y_min = min(lip_y_coords)
		y_max = max(lip_y_coords)

		# Вычисляем центр области
		center_x = (x_min + x_max) / 2
		center_y = (y_min + y_max) / 2

		# Добавляем margin
		width = x_max - x_min
		height = y_max - y_min

		base_width = width * (1 + 2 * LIP_MARGIN)
		base_height = height * (1 + 2 * LIP_MARGIN)

		current_aspect = base_width / base_height if base_height > 0 else TARGET_ASPECT_RATIO

		if current_aspect > TARGET_ASPECT_RATIO:
			# Слишком широкая область - увеличиваем высоту
			roi_width = base_width
			roi_height = base_width / TARGET_ASPECT_RATIO
		else:
			# Слишком высокая область - увеличиваем ширину
			roi_height = base_height
			roi_width = base_height * TARGET_ASPECT_RATIO

		# Вычисляем координаты ROI
		x_start = int(center_x - roi_width / 2)
		y_start = int(center_y - roi_height / 2)
		x_end = int(center_x + roi_width / 2)
		y_end = int(center_y + roi_height / 2)

		# Проверяем границы кадра и корректируем с сохранением пропорций
		frame_height, frame_width = frame.shape[:2]

		# Корректировка по X
		if x_start < 0:
			x_end -= x_start  # сдвигаем вправо
			x_start = 0
		if x_end > frame_width:
			x_start -= (x_end - frame_width)  # сдвигаем влево
			x_end = frame_width
			x_start = max(0, x_start)

		# Корректировка по Y
		if y_start < 0:
			y_end -= y_start  # сдвигаем вниз
			y_start = 0
		if y_end > frame_height:
			y_start -= (y_end - frame_height)  # сдвигаем вверх
			y_end = frame_height
			y_start = max(0, y_start)

		# Финальная проверка размеров
		actual_width = x_end - x_start
		actual_height = y_end - y_start

		if actual_width <= 0 or actual_height <= 0:
			print("Некорректные координаты ROI")
			return None

		# Вырезаем ROI
		mouth_roi = frame[y_start:y_end, x_start:x_end]

		# Проверяем, что ROI не пустая
		if mouth_roi.size == 0:
			print("Пустая область ROI")
			return None

		# Ресайзим до стандартного размера
		mouth_roi = cv2.resize(mouth_roi,  (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)  # ? scale

		return mouth_roi

	def process_videos(self):
		"""Обработка всех видео с учетом структуры"""
		# Создаем структуру каталогов для обработанных данных

		self.processed_data_path.mkdir(parents=True, exist_ok=True)
		video_files = list(self.video_path.glob("*.mp4"))

		if not video_files:
			print(f"Предупреждение: MP4 файлы не найдены в {self.video_path}")
			return

		video_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f.stem)) or 0))

		for video_file in tqdm(video_files):
			video_name = video_file.stem  # имя
			text_file = self.text_path / f"{video_name}.csv"
			video_processed_dir = self.processed_data_path / video_name # папка под видео
			video_processed_dir.mkdir(exist_ok=True)
			self.process_video(video_file, text_file, video_processed_dir)

	def process_video(self, video_path, text_file, output_dir):
		"""Обработка одного видеофайла"""

		df = pd.read_csv(text_file)
		cap = cv2.VideoCapture(video_path)

		fps = cap.get(cv2.CAP_PROP_FPS)
		total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		duration = total_frames / fps

		# Конвертируем временные метки в секунды
		timestamps = []
		for idx, row in df.iterrows():
			time_sec = time_to_sec(row['time'])
			phrase = row['text']
			timestamps.append({
				'start_sec': time_sec,
				'phrase': phrase,
				'index': idx
			})

		# Добавляем конечные точки (следующая фраза = конец предыдущей)
		for i in range(len(timestamps)):
			if i < len(timestamps) - 1:
				timestamps[i]['end_sec'] = timestamps[i + 1]['start_sec']
			else:
				timestamps[i]['end_sec'] = duration

		for ts in timestamps:
			start_sec = ts['start_sec']
			end_sec = ts['end_sec']
			phrase = ts['phrase']
			idx = ts['index']

			start_frame = int(start_sec * fps)
			end_frame = int(end_sec * fps)

			cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

			frames = []
			frame_count = 0
			current_frame = start_frame

			while current_frame < end_frame:
				ret, frame = cap.read()
				if not ret:
					break

				# Обрабатываем каждый N-й кадр для экономии ресурсов
				if frame_count % 2 == 0:  # Каждый второй кадр
					mouth_roi = self.extract_mouth_roi(frame)
					if mouth_roi is not None:
						frames.append(mouth_roi)

				frame_count += 1
				current_frame += 1

			encoded = self.tokenizer(
				phrase,
				padding=False,
				truncation=True,
				max_length=128,
				return_tensors=None
			)
			# Сохраняем обработанные кадры
			if frames:
				video_name = video_path.stem
				output_path = output_dir / f"{video_name}_{idx}.pkl"

				with open(output_path, 'wb') as f:
					pickle.dump({
						'frames': frames,
						'tokens': self.tokenizer.tokenize(phrase),
						'input_ids': encoded['input_ids'],
						'num_frames': len(frames)
					}, f)

				print(f"Обработано: {video_path.name}_{idx} -> {len(frames)} кадров")
			else:
				print(f"Предупреждение: Не удалось обработать {video_path.name}_{idx}")


class LipReading2Dataset(Dataset):
	def __init__(self, data_path, split='train_set', max_token_length=50, max_frames_length=90, target_frame_size=(64, 96)):
		self.data_path = Path(data_path) #/ split
		self.max_frames_length = max_frames_length
		self.max_token_length = max_token_length
		self.target_frame_size = target_frame_size

		self.files = list(self.data_path.glob("**/*.pkl"))
		self.files = sorted(self.files)
		print(f"Найдено {len(self.files)} файлов")

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		with open(self.files[idx], 'rb') as f:
			data = pickle.load(f)

		frames = data['frames']
		num_frames = data['num_frames']

		ids = data['input_ids']
		num_tokens = len(ids)

		frames = np.transpose(frames, (0, 3, 1, 2))

		# Padding/truncation кадров
		if num_frames > self.max_frames_length:
			indices = np.linspace(0, num_frames - 1, self.max_frames_length, dtype=int)
			frames = frames[indices]
			num_frames = self.max_frames_length
		elif num_frames < self.max_frames_length:
			padding = np.zeros((self.max_frames_length - num_frames, *frames.shape[1:]), dtype=np.float32)
			frames = np.concatenate([frames, padding], axis=0)

		return {
			'frames': torch.tensor(frames, dtype=torch.float32),
			'input_ids': torch.tensor(ids, dtype=torch.long),
			'num_frames': num_frames,
			'num_tokens': num_tokens,
			'phrase': data.get('tokens', '')
		}

def display_sample_subplots(dataset, sample_idx=0, max_frames=12, figsize=(12, 8), save_path = None):
	"""
    Отображает первый семпл из датасета с использованием subplots
    """
	# Получаем первый семпл из датасета
	sample = dataset[sample_idx]

	frames = sample['frames']
	num = sample['num_frames']
	if isinstance(frames, torch.Tensor):
		frames = frames.numpy()

	# (T, C, H, W) -> (T, H, W)
	if frames.ndim == 4:
		if frames.shape[1] == 1:  # Grayscale
			frames = frames[:, 0, :, :]
		else:  # RGB
			frames = frames[:, 0, :, :]

	show_frames = min(max_frames, num)

	phrase = sample['phrase']

	n_cols = 4
	n_rows = (show_frames + n_cols - 1) // n_cols

	fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

	# Делаем axes всегда 2D массивом
	if n_rows == 1:
		axes = axes.reshape(1, -1)

	# Заголовок с информацией
	title = f'Sample #{sample_idx}\n'
	title += f'Text: "{phrase[:50]}{"..." if len(phrase) > 50 else ""}"'
	fig.suptitle(title, fontsize=12)

	# Отображаем каждый кадр
	for i, ax in enumerate(axes.flat):
		if i < len(frames):
			ax.imshow(frames[i], cmap='gray')
		ax.axis('off')

	plt.tight_layout()

	if save_path:
		plt.savefig(save_path, dpi=150, bbox_inches='tight')
		print(f"Сохранено: {save_path}")
	else:
		plt.show()

	plt.close()


train_dataset = LipReading2Dataset(
    data_path='./Dataset_720_proc',
    max_token_length=50,
	max_frames_length=90
)

# Альтернативное отображение
display_sample_subplots(train_dataset, sample_idx=0, max_frames=12, figsize=(15, 9), save_path = './sample_720.png')
#print(train_dataset[0]['num_frames'], train_dataset[0]['num_tokens'])
#print(train_dataset[1]['num_frames'], train_dataset[1]['phrase'])
#print(train_dataset[2]['num_frames'], train_dataset[2]['num_tokens'])
#print(train_dataset[8]['num_frames'], train_dataset[8]['num_tokens'])



train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size

train_set, val_set = torch.utils.data.random_split(
    train_dataset, [train_size, val_size]
)

train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4)

#process_video("./dataset/video/pt1.mp4")


#preprocessor = LipReading2Preprocessor(
#    raw_data_path="./dataset_720",
#    processed_data_path="./Dataset_720_proc"
#)
#preprocessor.process_videos()
