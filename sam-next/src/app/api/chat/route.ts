import { NextRequest, NextResponse } from 'next/server'

// Use the exact OpenAI-compatible endpoint that works for free-tier LLaMA
const HF_URL = 'https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct/v1/chat/completions'

export async function POST(req: NextRequest) {
  const { message } = await req.json()
  if (!message?.trim()) return NextResponse.json({ error: 'Empty message' }, { status: 400 })

  const token = process.env.HF_TOKEN
  if (!token) return NextResponse.json({ error: 'Server misconfiguration' }, { status: 500 })

  try {
    const res = await fetch(HF_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'meta-llama/Llama-3.2-3B-Instruct',
        messages: [
          { role: 'system', content: 'You are SAM, a helpful medical AI assistant. Answer concisely and clearly in a friendly tone. Do not provide medical diagnoses.' },
          { role: 'user', content: message }
        ],
        max_tokens: 300,
        temperature: 0.4,
      }),
    })

    if (!res.ok) {
      const err = await res.text()
      console.error('HF API error:', res.status, err)
      if (res.status === 503 || err.includes('loading')) {
        return NextResponse.json({ reply: 'The AI model is currently loading into memory. Please try again in about 20 seconds!' })
      }
      return NextResponse.json({ error: 'AI service unavailable' }, { status: 502 })
    }

    const data = await res.json()
    const reply = data.choices?.[0]?.message?.content?.trim()
    return NextResponse.json({ reply: reply || 'No response generated.' })
  } catch (e: any) {
    console.error('Chat route error:', e.message || e)
    return NextResponse.json({ error: 'Internal error' }, { status: 500 })
  }
}
