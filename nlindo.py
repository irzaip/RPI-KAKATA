from __future__ import division

import re
import sys
import os
import io
import time
import audioop
import librosa
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
import pyaudio
from six.moves import queue
import subprocess
import gtts
import soundfile
import itertools

import numpy as np

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

#import listens

#ear = Listens()


class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""
    def __init__(self, rate, chunk, thres=800, waittime=2, debug=False, recfile=""):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True
        self.avg_rms = []
        self.thres = thres
        self.waittime = waittime
        self.time_start = time.time()
        self.time_end = time.time()
        self.debug = debug
        self.recorded = []
        self.recfile = recfile
        
    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1, rate=self._rate,
            input=True, frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        if self.recfile != "":
            soundfile.write(self.recfile,self.recorded,16000)

        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()
        #print(type(self.recorded))

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            #bdata is buffered data bytes converted to float
            bdata = librosa.util.buf_to_float(data[0], n_bytes=2)
            self.recorded.extend(bdata)
            rms = librosa.feature.rmse(S=bdata)
            rms = int(rms*32768)
            self.avg_rms.append(rms)

            if (self.thres < rms):
                self.time_start = time.time()
                
            self.time_end = time.time()
            pause = self.time_end - self.time_start 
            if ( pause > self.waittime):
                print("Exit..")
                print(len(self.recorded))
                print(self.recorded[0])
                break
                
            if self.debug:
                print('{0:.2f}'.format(float(pause)), end='',flush=False)
                print("_",end='',flush=False)
                
            yield b''.join(data)


def listen_print_loop(responses):
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if result.is_final:
            #print(transcript)
            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r'\b(exit|quit)\b', transcript, re.I):
                print('Exiting..')
                break

            num_chars_printed = 0
            return transcript

def dengar(thres=800,waittime=2, debug=False, recfile="", transcribe=True):
    """Mendengarkan menggunakan microfon waittime adalah waktu kosong setelah bicara,
    recfile rekam kedalam format wav 16000"""
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = 'id-ID'  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code)
    streaming_config = types.StreamingRecognitionConfig(
        config=config,
        interim_results=True)

    
    with MicrophoneStream(RATE, CHUNK, thres=thres, waittime=waittime, debug=debug, recfile=recfile) as stream:
        audio_generator = stream.generator()
        
        if transcribe:
            requests = (types.StreamingRecognizeRequest(audio_content=content)
                        for content in audio_generator)
        
            responses = client.streaming_recognize(streaming_config, requests)
    
        # Now, put the transcription responses to use.
            transcript = listen_print_loop(responses)
        else:
            for content in audio_generator:
                pass
            transcript=""

    return transcript

def katakan(sentence, slow = False, recfile="", playaudio=True, to_wav=False):
    """mengatakan apa yang diminta, bisa di rekam ke file .mp3 24000 hz
    - converter ke wav belum jalan"""
    try:
      rets = gtts.gTTS(sentence, lang="id", slow = slow)
    except:
        print("Error calling gTTS -- quit")

    if playaudio:
        try:
            rets.save("_tempspeech.mp3")
            subprocess.call(["ffplay", "-nodisp", "-autoexit", "_tempspeech.mp3"])
        except:
            print("Error calling - ffplay --maybe ffmpeg not installed correctly?")
    
        try:
            os.remove("_tempspeech.mp3")
        except:
            pass

    if recfile != "":
        try:
            rets.save(recfile)
        except:
            print("Error saving file-",recfile)
        if to_wav:
            try:
                #subprocess.call(["ffmpeg","-i","geregete.wav","-f","s16le","-ar","16000","-y","medi.wav"])    
                print("not working wav conversion")
            except:
                print("Error converting mp3 to wav - maybe ffmpeg not installed?")    



def transcrip(speech_file):
    """Mentranskrip dari sebuah audio file -wav 16000 mono"""
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    client = speech.SpeechClient()

    with io.open(speech_file, 'rb') as audio_file:
        content = audio_file.read()

    audio = types.RecognitionAudio(content=content)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='id-ID')

    response = client.recognize(config, audio)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        transkrip=result.alternatives[0].transcript
    #return
    return transkrip

def plotfile(filename):
    import matplotlib.pyplot as plt
    import soundfile
    import numpy as np
    audiodata = soundfile.read(filename)
    a= np.array(audiodata[0])
    plt.plot(a)
    print(a.shape)



def fliparr(arr):
    """Flip - reverse an ndarray"""
    return np.fliplr([arr])[0]

def clean_file(filename,step=160,thres=540,resfile=""):
    """bersihkan bagian awal dan akhir yang dibawah threshold, pengambilan data berdasarkan step 
    dan akhirnya di save ke resfile. apabila kosong, akan menimpa yang lama"""
    arr, _ = soundfile.read(filename)
    #arr = fliparr(arr)
    rg = 0
    bstep=0
    estep=step

    for i in range(40):
        carr = librosa.feature.rmse(S=arr[bstep:estep]) * 32768    
        print(carr)
        
        if carr < thres or np.isnan(carr):
            rg += 1
            bstep = bstep + step
            estep = estep + step
        else:
            break

    #balik arraynya dan mulai lagi dari awal.        
    arr=arr[estep:-1]
    arr = fliparr(arr)
    rg = 0
    bstep=0
    estep=step
    
    for i in range(40):
        carr = librosa.feature.rmse(S=arr[bstep:estep]) * 32768    
        print(carr)
        
        if carr < thres or np.isnan(carr):
            rg += 1
            bstep = bstep + step
            estep = estep + step
        else:
            break
        
    arr=arr[estep:-1]
    arr = fliparr(arr)
    if resfile != "":
        soundfile.write(resfile, arr, 16000)
    else:
        soundfile.write(filename,arr, 16000)
        

if __name__ == '__main__':
    print(dengar(thres=800,waittime=3))