import { useState, useRef, useCallback, useEffect } from 'react'
import { Mic, Square } from 'lucide-react'

const API_BASE = ''

function App() {
  const [isRecording, setIsRecording] = useState(false)
  const [transcript, setTranscript] = useState('')
  const [language, setLanguage] = useState('')
  const [status, setStatus] = useState<'idle' | 'recording' | 'error'>('idle')
  const [errorMsg, setErrorMsg] = useState('')

  const sessionIdRef = useRef<string | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const bufferRef = useRef<Float32Array>(new Float32Array(0))
  const pushingRef = useRef(false)
  const recordingRef = useRef(false)
  const transcriptRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight
    }
  }, [transcript])

  const TARGET_SR = 16000
  const CHUNK_MS = 500

  const concatFloat32 = (a: Float32Array, b: Float32Array): Float32Array => {
    const out = new Float32Array(a.length + b.length)
    out.set(a, 0)
    out.set(b, a.length)
    return out
  }

  const resampleLinear = (input: Float32Array, srcSr: number, dstSr: number): Float32Array => {
    if (srcSr === dstSr) return input
    const ratio = dstSr / srcSr
    const outLen = Math.max(0, Math.round(input.length * ratio))
    const out = new Float32Array(outLen)
    for (let i = 0; i < outLen; i++) {
      const x = i / ratio
      const x0 = Math.floor(x)
      const x1 = Math.min(x0 + 1, input.length - 1)
      const t = x - x0
      out[i] = input[x0] * (1 - t) + input[x1] * t
    }
    return out
  }

  const apiStart = async (): Promise<string> => {
    const r = await fetch(`${API_BASE}/api/start`, { method: 'POST' })
    if (!r.ok) throw new Error(await r.text())
    const j = await r.json()
    return j.session_id
  }

  const apiPushChunk = async (sessionId: string, float32_16k: Float32Array) => {
    // Create a copy of the buffer to ensure it's a regular ArrayBuffer
    const buffer = new ArrayBuffer(float32_16k.byteLength)
    new Float32Array(buffer).set(float32_16k)

    const r = await fetch(`${API_BASE}/api/chunk?session_id=${encodeURIComponent(sessionId)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/octet-stream' },
      body: buffer,
    })
    if (!r.ok) throw new Error(await r.text())
    return await r.json()
  }

  const apiFinish = async (sessionId: string) => {
    const r = await fetch(`${API_BASE}/api/finish?session_id=${encodeURIComponent(sessionId)}`, {
      method: 'POST',
    })
    if (!r.ok) throw new Error(await r.text())
    return await r.json()
  }

  const pump = useCallback(async () => {
    if (pushingRef.current) return
    pushingRef.current = true

    const chunkSamples = Math.round(TARGET_SR * (CHUNK_MS / 1000))

    try {
      while (recordingRef.current && bufferRef.current.length >= chunkSamples) {
        const chunk = bufferRef.current.slice(0, chunkSamples)
        bufferRef.current = bufferRef.current.slice(chunkSamples)

        if (sessionIdRef.current) {
          const j = await apiPushChunk(sessionIdRef.current, chunk)
          setLanguage(j.language || '')
          setTranscript(j.text || '')
        }
      }
    } catch (err) {
      console.error(err)
      setErrorMsg(`Backend error: ${err}`)
      setStatus('error')
    } finally {
      pushingRef.current = false
    }
  }, [])

  const stopAudioPipeline = async () => {
    try {
      if (processorRef.current) {
        processorRef.current.disconnect()
        processorRef.current.onaudioprocess = null
      }
      if (sourceRef.current) sourceRef.current.disconnect()
      if (audioContextRef.current) await audioContextRef.current.close()
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((t) => t.stop())
      }
    } catch (e) {
      console.error(e)
    }
    processorRef.current = null
    sourceRef.current = null
    audioContextRef.current = null
    mediaStreamRef.current = null
  }

  const handleStart = async () => {
    if (isRecording) return

    setTranscript('')
    setLanguage('')
    setErrorMsg('')
    bufferRef.current = new Float32Array(0)

    try {
      setStatus('recording')
      setIsRecording(true)
      recordingRef.current = true

      sessionIdRef.current = await apiStart()

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      })
      mediaStreamRef.current = stream

      const audioCtx = new AudioContext()
      audioContextRef.current = audioCtx

      const source = audioCtx.createMediaStreamSource(stream)
      sourceRef.current = source

      const processor = audioCtx.createScriptProcessor(4096, 1, 1)
      processorRef.current = processor

      processor.onaudioprocess = (e) => {
        if (!recordingRef.current) return
        const input = e.inputBuffer.getChannelData(0)
        const resampled = resampleLinear(new Float32Array(input), audioCtx.sampleRate, TARGET_SR)
        bufferRef.current = concatFloat32(bufferRef.current, resampled)
        if (!pushingRef.current) pump()
      }

      source.connect(processor)
      processor.connect(audioCtx.destination)
    } catch (err) {
      console.error(err)
      setErrorMsg(`Start failed: ${err}`)
      setStatus('error')
      setIsRecording(false)
      recordingRef.current = false
      sessionIdRef.current = null
      await stopAudioPipeline()
    }
  }

  const handleStop = async () => {
    if (!isRecording) return

    recordingRef.current = false
    setIsRecording(false)

    await stopAudioPipeline()

    try {
      if (sessionIdRef.current) {
        const j = await apiFinish(sessionIdRef.current)
        setLanguage(j.language || '')
        setTranscript(j.text || '')
      }
      setStatus('idle')
    } catch (err) {
      console.error(err)
      setErrorMsg(`Finish failed: ${err}`)
      setStatus('error')
    } finally {
      sessionIdRef.current = null
      bufferRef.current = new Float32Array(0)
      pushingRef.current = false
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-2xl mx-auto">
        <div className="bg-white rounded-xl shadow-lg p-6 space-y-6">
          <h1 className="text-2xl font-bold text-gray-900">
            Qwen3-ASR Streaming Demo
          </h1>

          {/* Controls */}
          <div className="flex gap-4">
            <button
              onClick={handleStart}
              disabled={isRecording}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors ${
                isRecording
                  ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  : 'bg-green-600 text-white hover:bg-green-700'
              }`}
            >
              <Mic className="w-5 h-5" />
              Start
            </button>

            <button
              onClick={handleStop}
              disabled={!isRecording}
              className={`flex items-center gap-2 px-6 py-3 rounded-lg font-medium transition-colors ${
                !isRecording
                  ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  : 'bg-red-600 text-white hover:bg-red-700'
              }`}
            >
              <Square className="w-5 h-5" />
              Stop
            </button>
          </div>

          {/* Status */}
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-500">Status:</span>
            <span
              className={`px-3 py-1 rounded-full text-sm font-medium ${
                status === 'idle'
                  ? 'bg-gray-100 text-gray-600'
                  : status === 'recording'
                  ? 'bg-green-100 text-green-700'
                  : 'bg-red-100 text-red-700'
              }`}
            >
              {status === 'idle' && 'Idle'}
              {status === 'recording' && 'Recording...'}
              {status === 'error' && 'Error'}
            </span>
            {isRecording && (
              <span className="flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-3 w-3 rounded-full bg-red-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
              </span>
            )}
          </div>

          {/* Language */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Detected Language
            </label>
            <div className="px-4 py-2 bg-gray-50 rounded-lg border border-gray-200 font-mono text-sm">
              {language || '—'}
            </div>
          </div>

          {/* Transcript */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Transcript
            </label>
            <textarea
              ref={transcriptRef}
              readOnly
              value={transcript}
              placeholder="Transcription will appear here..."
              className="w-full h-96 px-4 py-3 bg-gray-50 rounded-lg border border-gray-200 resize-none font-mono text-sm focus:outline-none"
            />
          </div>

          {/* Error */}
          {errorMsg && (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
              {errorMsg}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
