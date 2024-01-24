import marimo

__generated_with = "0.1.81"
app = marimo.App()


@app.cell(hide_code=True)
def __(mo):
    # upload_button = mo.ui.file(kind="button")
    audiopath = mo.ui.text(placeholder="path/your/audio...")
    savefolder = mo.ui.text(placeholder="path/your/saveFolder...")
    speakername = mo.ui.text(placeholder="speaker name...")

    start_button = mo.ui.button(
        label="start split",
        on_change=lambda _: print("start"),
    )
    formats_choose = mo.ui.dropdown(
        options=dict(
            sorted(
                {
                    "GPT-SoVITS": "GPT-SoVITS",
                }.items()
            )
        ),
    )

    format_chooser = mo.md(
        f"""
        Data format: {formats_choose}
        """
    )
    modal_choose = mo.ui.dropdown(
        options=dict(
            sorted(
                {
                    "LargeV3(most powerful)": "openai/whisper-large-v3",
                    # "distil-LargeV2(more faster)": "distil-whisper/large-v2",
                }.items()
            )
        ),
    )

    modal_chooser = mo.md(
        f"""
        Whisper modal: {modal_choose}
        """
    )
    lang_choose = mo.ui.dropdown(
        options=dict(
            sorted(
                {
                    "Chinese": "zh",
                    "English": "en",
                    "Japanese": "ja",
                }.items()
            )
        ),
    )

    lang_chooser = mo.md(
        f"""
        Language: {lang_choose}
        """
    )
    vram_choose = mo.ui.dropdown(
        options=dict(
            sorted(
                {
                    "> 8GB": True,
                    "≤ 8GB": False,
                }.items()
            )
        ),
    )

    vram_chooser = mo.md(
        f"""
        VRam size: {vram_choose}
        """
    )
    mo.hstack(
        [
            [audiopath, savefolder, speakername],
            [format_chooser, modal_chooser, lang_chooser, vram_chooser],
        ],
        justify="start",
    )
    return (
        audiopath,
        format_chooser,
        formats_choose,
        lang_choose,
        lang_chooser,
        modal_choose,
        modal_chooser,
        savefolder,
        speakername,
        start_button,
        vram_choose,
        vram_chooser,
    )


@app.cell(hide_code=True)
def __(main, mo, set_done_msg, set_warn_elem):
    set_done_msg("")
    set_warn_elem("")
    start_split = mo.ui.button(
        label="Link Start!",
        on_change=lambda _: main(),
    )

    mo.hstack([start_split], justify="center")
    return start_split,


@app.cell(hide_code=True)
def __(get_done_msg):
    done_msg = ""
    if get_done_msg():
        done_msg = get_done_msg()

    done_msg
    return done_msg,


@app.cell(hide_code=True)
def __(get_warn_elem, mo):
    if get_warn_elem():
        warn_elem = mo.md(
            f"""
            **Invaild value! You must choose it!**

            {get_warn_elem()}
            """
        ).callout(kind="warn")
    else:
        warn_elem = ""
    mo.hstack([warn_elem], "center")
    return warn_elem,


@app.cell(hide_code=True)
def __(mo, torch):
    show_notif = ""
    if not torch.cuda.is_available():
        show_notif = mo.md(
            """
        **CUDA not available!**

        Check your conda environment and install cudatoolkit, then install torch corresponding to the cuda version down from PyTorch Website.

        https://pytorch.org/get-started/locally/
        """
        ).callout(kind="warn")
    show_notif
    return show_notif,


@app.cell
def __(Path, audiopath, savefolder):
    audio_path = Path(audiopath.value)
    save_path = Path(savefolder.value)
    return audio_path, save_path


@app.cell
def __():
    import torch
    import subprocess
    import marimo as mo
    import toml
    import os
    import sys
    import spacy
    import librosa
    import numpy as np
    from mdx23.inference import start
    import shutil
    # from voicefixer import VoiceFixer
    from pathlib import Path
    from pydub import AudioSegment
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import soundfile as sf
    from transformers import pipeline
    return (
        AudioSegment,
        Path,
        ThreadPoolExecutor,
        as_completed,
        librosa,
        mo,
        np,
        os,
        pipeline,
        sf,
        shutil,
        spacy,
        start,
        subprocess,
        sys,
        toml,
        torch,
    )


@app.cell
def __(sys):
    python_executable = sys.executable
    return python_executable,


@app.cell
def __(
    ThreadPoolExecutor,
    as_completed,
    audio_path,
    audiopath,
    extract_vocal,
    formats_choose,
    lang_choose,
    mo,
    modal_choose,
    os,
    processing_segment,
    save_path,
    savefolder,
    shutil,
    speakername,
    spliter,
    transcribe_with_whisper,
    vram_choose,
):
    get_warn_elem, set_warn_elem = mo.state("")
    get_done_msg, set_done_msg = mo.state("")
    def main():
        invaild_value = []
        if not audiopath.value:
            invaild_value.append("no audio path")
        if not savefolder.value:
            invaild_value.append("no save folder")
        if not speakername.value:
            invaild_value.append("no speaker name")
        if not formats_choose.value:
            invaild_value.append("no format choose")
        if not modal_choose.value:
            invaild_value.append("no modal choose")
        if not lang_choose.value:
            invaild_value.append("no language choose")
        if vram_choose.value == None or vram_choose.value == "":
            invaild_value.append("no choose vram")

        if len(invaild_value) != 0:
            set_warn_elem(invaild_value)
            raise ValueError(f"{invaild_value}")

        # check file exiests!
        if not os.path.exists(audio_path):
            raise ValueError(f"File [{audio_path}] does not exist!!")

        with mo.status.spinner(title="clean vocal...") as _spinner:
            extract_vocal()
            _spinner.update(title="transcribe with whisper...")
            transcribe_with_whisper()
            _spinner.update(title="split segments...")
            audio, new_sentences = spliter()
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(processing_segment, audio, idx, segment)
                for idx, segment in enumerate(new_sentences)
            ]
            progress_bar = mo.status.progress_bar(range(len(futures)))
            results = [
                future.result()
                for _, future in zip(progress_bar, as_completed(futures))
            ]
        if not os.path.isabs(save_path):
            current_directory = os.getcwd()
            full_path = os.path.join(current_directory, save_path, "segments.list")
            data_folder = os.path.join(current_directory, save_path, "final_segments")
        else:
            full_path = os.path.join(save_path, "segments.list")
            data_folder = os.path.join(save_path, "final_segments")
        shutil.rmtree(f"{save_path}/clean_vocal")
        shutil.rmtree(f"{save_path}/whisper_segments")
        done_msg = mo.md(
            f"## Your data is ready! \n\nAnnotation path: `{full_path}`\n\nData folder: `{data_folder}`\n\nRun cell above for another audio."
        )
        set_done_msg(done_msg)
        # progress_bar = mo.status.progress_bar(range(len(new_sentences)))
        # for _, (idx, segment) in zip(progress_bar, enumerate(new_sentences)):
        #     processing_segment(audio, idx, segment)
    return get_done_msg, get_warn_elem, main, set_done_msg, set_warn_elem


@app.cell
def __(audio_path, os, save_path, start, vram_choose):
    def extract_vocal():
        os.makedirs(f"{save_path}/clean_vocal", exist_ok=True)
        arg = {
            "input_audio": [
                audio_path,
            ],
            "output_folder": f"{save_path}/clean_vocal",
            "cpu": False,
            "overlap_demucs": 0.1,
            "overlap_VOCFT": 0.1,
            "overlap_VitLarge": 1,
            "overlap_InstVoc": 1,
            "weight_InstVoc": 8,
            "weight_VOCFT": 1,
            "weight_VitLarge": 5,
            "single_onnx": False,
            "large_gpu": vram_choose.value,
            "BigShifts": 7,
            "vocals_only": True,
            "use_VOCFT": False,
            "output_format": "FLOAT",
        }
        start(arg)
        # resemble_enhance in_dir out_dir --denoise_only
        # resemble_enhance 对那种本身就含混的语音会变得奇怪
        # filename_stem = audio_path.stem
        # voicefixer = VoiceFixer() # bad effect
        # voicefixer.restore(
        #     input=f"{save_path}/clean_vocal/{filename_stem}_vocals.wav",
        #     output=f"{save_path}/clean_vocal/vocals_clean.wav",
        #     cuda=True,
        #     mode=0,
        # )
    return extract_vocal,


@app.cell
def __(audio_path, modal_choose, pipeline, save_path, toml, torch):
    def transcribe_with_whisper():
        print("start transcribe")
        pipe = pipeline(
            "automatic-speech-recognition",
            model=modal_choose.value,
            torch_dtype=torch.float16,
            device="cuda",  # or mps for Mac devices
            model_kwargs={"use_flash_attention_2": False},
        )
        filename_stem = audio_path.stem
        outputs = pipe(
            f"{save_path}/clean_vocal/{filename_stem}_vocals.wav",
            chunk_length_s=30,
            batch_size=1,
            return_timestamps="word",
            generate_kwargs={"task": "transcribe", "language": None},
        )
        outputs["text"] = outputs["text"].strip()
        with open(
            f"{save_path}/transcribe_outputs.toml", "w", encoding="utf-8"
        ) as f:
            toml.dump(outputs, f)
    return transcribe_with_whisper,


@app.cell
def __(
    AudioSegment,
    audio_path,
    lang_choose,
    python_executable,
    save_path,
    spacy,
    subprocess,
    toml,
):
    def spliter():
        outputs = toml.load(f"{save_path}/transcribe_outputs.toml")
        if lang_choose.value == "en":
            try:
                nlp = spacy.load("en_core_web_sm")
            except:
                subprocess.run(
                    [
                        python_executable,
                        "-m",
                        "spacy",
                        "download",
                        "en_core_web_sm",
                    ]
                )
                nlp = spacy.load("en_core_web_sm")
        elif lang_choose.value == "zh":
            try:
                nlp = spacy.load("zh_core_web_sm")
            except:
                subprocess.run(
                    [
                        python_executable,
                        "-m",
                        "spacy",
                        "download",
                        "zh_core_web_sm",
                    ]
                )
                nlp = spacy.load("zh_core_web_sm")
        elif lang_choose.value == "ja":
            try:
                nlp = spacy.load("ja_core_news_sm")
            except:
                subprocess.run(
                    [
                        python_executable,
                        "-m",
                        "spacy",
                        "download",
                        "ja_core_news_sm",
                    ]
                )
                nlp = spacy.load("ja_core_news_sm")
        else:
            raise ValueError("no language choose")

        doc = nlp(outputs["text"])
        sentences = [sent.text for sent in doc.sents]
        # [print(sent) for sent in sentences]
        new_chunks = []
        for chunk in outputs["chunks"]:
            if chunk["text"].startswith("-"):
                new_chunks[-1]["text"] += chunk["text"]

                new_chunks[-1]["timestamp"][-1] = chunk["timestamp"][-1]
            else:
                new_chunks.append(chunk)
                new_chunks[-1]["text"] = chunk["text"].strip()
        # print(new_chunks)
        new_sentences = []

        for sentence in sentences:
            if lang_choose.value == "en":
                sent = sentence.split()
                # print(sent)
                idx = sent.index(sent[-1])
                sent_chunk = [new_chunks[idx] for idx in range(idx + 1)]
                new_chunks = new_chunks[idx + 1 :]
            elif lang_choose.value == "ja":
                sent_chunk = []
                sentence_len = len(sentence)
                for idx, chunk in enumerate(new_chunks):
                    if chunk["text"] in sentence and sentence_len > 0:
                        sent_chunk.append(chunk)
                        sentence_len -= len(chunk["text"])
                    elif sentence_len != 0:
                        sent_chunk = []
                        sentence_len = len(sentence)
                    else:
                        new_chunks = new_chunks[idx:]
                        break
            # print(idx, sent_chunk)
            new_sentences.append(
                {
                    "text": sentence,
                    "start": sent_chunk[0]["timestamp"][0],
                    "end": sent_chunk[-1]["timestamp"][-1],
                    "chunks": sent_chunk,
                }
            )
        filename_stem = audio_path.stem
        audio = AudioSegment.from_file(
            f"{save_path}/clean_vocal/{filename_stem}_vocals.wav"
        )

        return audio, new_sentences
    return spliter,


@app.cell
def __(
    formats_choose,
    lang_choose,
    librosa,
    np,
    os,
    save_path,
    sf,
    speakername,
):
    def processing_segment(audio, idx, segment):
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        padding_duration_ms = 1000
        ps = padding_duration_ms
        pe = padding_duration_ms
        final_save_path = f"{save_path}/final_segments"
        whisper_seg_save_path = f"{save_path}/whisper_segments"

        os.makedirs(final_save_path, exist_ok=True)

        os.makedirs(whisper_seg_save_path, exist_ok=True)
        segment_audio = audio[start_ms:end_ms]

        # vocal_path|speaker_name|language|text
        if formats_choose.value == "GPT-SoVITS":
            entry = f"{final_save_path}/segment_{idx}.wav|{speakername.value}|{lang_choose.value}|{segment['text']}\n"
        # audio_byte_stream = io.BytesIO()
        segment_audio.export(
            f"{whisper_seg_save_path}/segment_{idx}.wav", format="wav"
        )
        if start_ms - padding_duration_ms < 0:
            ps = start_ms
        temp_audio = audio[start_ms - ps : end_ms + pe]
        temp_audio.export(
            f"{whisper_seg_save_path}/segment_len_{idx}.wav", format="wav"
        )

        ay, sr = librosa.load(
            f"{whisper_seg_save_path}/segment_{idx}.wav", sr=None
        )
        if len(ay) == 0:
            print(f"segment_{idx} is empty")
            return

        ty, tsr = librosa.load(
            f"{whisper_seg_save_path}/segment_len_{idx}.wav", sr=None
        )

        frame_length = 2048
        hop_length = 512

        ps_samples = int(sr * ps / 1000)
        pe_samples = int(sr * pe / 1000)

        silence_before = np.zeros(ps_samples)
        silence_after = np.zeros(pe_samples)

        y = np.concatenate((silence_before, ay, silence_after))

        ay_start = len(silence_before) - 1 if len(silence_before) != 0 else 0

        ay_end = len(y) - len(silence_after) - 2

        energy = np.array(
            [
                np.sum(np.abs(y[i : i + frame_length] ** 2))
                for i in range(0, len(y), hop_length)
            ]
        )
        temp_energy = np.array(
            [
                np.sum(np.abs(ty[i : i + frame_length] ** 2))
                for i in range(0, len(ty), hop_length)
            ]
        )

        def get_envelope(y, sr):
            if len(y) != 0:
                window_size = int(sr * 0.02)

                window = np.ones(window_size) / window_size

                envelope = np.convolve(np.abs(y), window, mode="same")
                return envelope
            else:
                return []

        y_elope = get_envelope(y, sr)
        y_elope = get_envelope(y_elope, sr)
        y_elope = get_envelope(y_elope, sr)

        ty_elope = get_envelope(ty, tsr)
        sen_env = get_envelope(ty_elope, tsr)
        trd_env = get_envelope(sen_env, tsr)

        m = [
            trd_env[idx]
            if trd_env[idx] - trd_env[idx - 1] > 0
            and trd_env[idx] - trd_env[idx + 1] > 0
            else 0
            for idx in range(len(trd_env) - 1)
        ]
        m = [i if i > 0.005 else 0 for i in m]

        n = [
            trd_env[idx] + 0.1
            if trd_env[idx] - trd_env[idx - 1] <= 0
            and trd_env[idx] - trd_env[idx + 1] <= 0
            else 0
            for idx in range(len(trd_env) - 1)
        ]
        n = [round(i, 3) for i in n]
        # print([i for i in n if i != 0 and i != 0.1])
        # print("n", len(n), "aystart", ay_start, "ayend", ay_end)
        countx = 0
        index2m = 0
        last_low_idx = 0

        def get_whole(trd_env):
            new_list = [0] * len(trd_env)
            for idx in range(1, len(m) - 1):
                if m[idx] != 0:
                    startx = idx
                    while startx > 0 and n[startx] == 0:
                        startx -= 1
                    endx = idx
                    while endx < len(n) - 1 and n[endx] == 0:
                        endx += 1

                    for i in range(startx, endx + 1):
                        new_list[i] = m[idx]
            return new_list

        new_list = get_whole(trd_env)

        for idxz, elem in enumerate(new_list[ay_end:], start=ay_end):
            if elem == 0:
                new_list[idxz:] = [0] * (len(new_list) - idxz)

        last_value = 0
        start_value = new_list[ay_start]
        end_value = new_list[ay_end]
        sen_set = False
        # for idxz, i in enumerate(new_list):
        #     if idxz > ay_start and idxz <= ay_end - (ay_end - ay_start) / 2:
        #         # if abs(i - start_value) < 0.015:
        #         if i == start_value:
        #             new2_list[idxz] = 0
        #         else:
        #             new2_list[idxz] = i

        #     elif idxz > ay_end:
        #         if abs(i - end_value) < 0.015 and abs(i - last_value) < 0.015:
        #             new2_list[idxz] = i
        #         else:
        #             new2_list[idxz] = 0
        #             end_value = -1
        #     elif idxz > ay_start and idxz <= ay_end:
        #         new2_list[idxz] = i
        #     else:
        #         new2_list[idxz] = 0

        #     last_value = i

        cut_start = 0
        right_low = 0
        right_high = 0
        for ix, i in enumerate(n):
            if n[ay_start - 1] == 0.1 and n[ay_start + 1] == 0.1:
                if ix > ay_start and i != 0.1:
                    cut_start = ix
                    break
            elif ix > ay_start and i != 0:
                right_low = ix
                for ix, i in enumerate(m):
                    if ix > ay_start and i != 0:
                        right_high = ix
                        break
                if right_low > right_high:
                    for idxz, elem in reversed(list(enumerate(n[:ay_start]))):
                        if elem != 0:
                            cut_start = idxz
                            break
                else:
                    if n[right_low] == n[right_low + 1]:
                        for idxz, elem in enumerate(
                            n[right_low:], start=right_low
                        ):
                            if elem != n[right_low]:
                                right_low = idxz
                                break
                    cut_start = right_low
                break
        for idxz, elem in reversed(list(enumerate(n[:cut_start]))):
            if elem != 0 and idxz != cut_start:
                cut_start = idxz if elem < n[cut_start] else cut_start
                break

        min_value = n[cut_start]
        min_value_index = cut_start

        for i in range(cut_start + 1, len(n)):
            if n[i] <= min_value and n[i] != 0:
                min_value = n[i]
                min_value_index = i

            elif n[i] > min_value and n[i] != 0:
                break

        cut_start = min_value_index

        if new_list[cut_start + 1] == 0:
            for idxz, elem in enumerate(
                new_list[cut_start + 1 :], start=cut_start + 1
            ):
                if elem != 0:
                    cut_start = idxz
                    break

        cut_end = 0
        right_low = 0
        right_high = 0
        after = ay_end + 1 if ay_end + 1 < len(n) else len(n) - 1
        for ix, i in enumerate(n):
            if n[ay_end - 1] == 0.1 and n[after] == 0.1:
                if ix > ay_end and i != 0.1:
                    cut_end = ix
                    break
            elif ix > ay_end and i != 0:
                right_low = ix
                cut_end = right_low
                # print(cut_end)
                for ix, i in enumerate(m):
                    if ix > ay_end and i != 0 and i < 0.12:
                        right_high = ix
                        break
                if right_high == 0:
                    right_high = right_low + 1
                if right_low > right_high:
                    for elem in reversed(n[:ay_end]):
                        if elem != 0:
                            cut_end = n.index(elem)
                            break
                else:
                    for idxz, elem in reversed(list(enumerate(n[:right_high]))):
                        if elem != 0:
                            cut_end = idxz
                            break
            else:
                for idxz, elem in reversed(list(enumerate(n[:ay_end]))):
                    if elem != 0:
                        cut_end = idxz
                        break
                break

        min_value = n[cut_end]
        min_value_index = cut_end

        increase_count = 0

        last_increase_value = min_value

        for i in range(cut_end + 1, len(n)):
            if n[i] <= min_value and n[i] != 0:
                min_value = n[i]
                min_value_index = i
                increase_count = 0
                # last_increase_value = min_value
                # print(min_value_index, min_value)
            elif (
                n[i] > min_value
                and n[i] != 0
                and n[i] < 0.12
                and new_list[ix] < 0.02
            ):
                increase_count += 1
                if increase_count > 1:
                    break
        cut_end = min_value_index

        if new_list[cut_end - 1] == 0:
            for idxz, elem in reversed(list(enumerate(new_list[: cut_end - 1]))):
                if elem != 0:
                    cut_end = idxz
                    break

        if cut_start > cut_end:
            value = new_list[ay_start]

            if value == 0:
                for idxz, elem in enumerate(new_list[ay_start:], start=ay_start):
                    if elem != 0:
                        cut_start = idxz
                        break
            else:
                for idxz, elem in reversed(
                    list(enumerate(new_list[: ay_start - 1]))
                ):
                    if elem != value:
                        cut_start = idxz
                        break

        if cut_end < cut_start:
            value = new_list[ay_end]
            if value == 0:
                for idxz, elem in reversed(list(enumerate(new_list[:ay_end]))):
                    if elem != 0:
                        cut_end = idxz
                        break
            else:
                for idxz, elem in enumerate(new_list[ay_end:], start=ay_end):
                    if elem != value:
                        cut_end = idxz
                        break

        final_audio = []
        for ix, i in enumerate(ty):
            if ix >= cut_start and ix <= cut_end:
                final_audio.append(i)

        sf.write(f"{final_save_path}/segment_{idx}.wav", final_audio, tsr)
        with open(f"{save_path}/segments.list", "a", encoding="utf-8") as file:
            file.write(entry)
    return processing_segment,


if __name__ == "__main__":
    app.run()
