import librosa
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Carregando o arquivo de áudio
    audio_path = 'worldop.mp3'
    y, sr = librosa.load(audio_path)

    # Aplicando o STFT
    D = librosa.stft(y)

    # Obtendo o espectrograma em dB
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Plotando o espectrograma
    plt.figure(figsize=(12, 6))
    librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma do áudio')
    plt.show()
