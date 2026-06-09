from SDP import Pipeline

SAMPLING_RATE = 16000

if __name__ == "__main__":
    p =  Pipeline()
    audio_file = "examples/part1/bacsidatnhkhoavitadoc_1.wav"
    p.forward(audio_file, sampling_rate=SAMPLING_RATE)
    