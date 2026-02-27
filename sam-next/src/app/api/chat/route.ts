import { NextRequest, NextResponse } from 'next/server'

const HF_URL = 'https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2'

export async function POST(req: NextRequest) {
  const { message } = await req.json()
  if (!message?.trim()) return NextResponse.json({ error: 'Empty message' }, { status: 400 })

  const token = process.env.HF_TOKEN
  if (!token) return NextResponse.json({ error: 'Server misconfiguration' }, { status: 500 })

  const prompt = `<s>[INST] You are SAM, a helpful medical AI assistant. Answer concisely and clearly. User question: ${message} [/INST]`

  try {
    const res = await fetch(HF_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        inputs: prompt,
        parameters: { max_new_tokens: 300, temperature: 0.4, return_full_text: false },
      }),
    })

    if (!res.ok) {
      const err = await res.text()
      if (res.status === 503) {
        return NextResponse.json({ reply: 'The AI model is loading, please try again in a moment.' })
      }
      console.error('HF API error:', err)
      return NextResponse.json({ error: 'AI service unavailable' }, { status: 502 })
    }

    const data = await res.json()
    const reply = Array.isArray(data) ? data[0]?.generated_text?.trim() : data?.generated_text?.trim()
    return NextResponse.json({ reply: reply || 'No response generated.' })
  } catch (e) {
    console.error('Chat route error:', e)
    return NextResponse.json({ error: 'Internal error' }, { status: 500 })
  }
}
