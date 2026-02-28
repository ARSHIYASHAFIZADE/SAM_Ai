import { NextRequest, NextResponse } from 'next/server'

const HF_URL = 'https://router.huggingface.co/hf-inference/v1/chat/completions'

export async function POST(req: NextRequest) {
  try {
    const { message } = await req.json()
    if (!message?.trim()) {
      console.error('Chat error: empty message string')
      return NextResponse.json({ error: 'Empty message' }, { status: 400 })
    }

    const token = process.env.HF_TOKEN
    if (!token) {
      console.error('Chat error: missing HF_TOKEN from environment variables.')
      return NextResponse.json({ error: 'Server misconfiguration' }, { status: 500 })
    }

    console.log(`Sending request to Hugging Face with message: "${message.substring(0, 50)}..."`)

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

    console.log(`Received HF response. Status = ${res.status}`)

    if (!res.ok) {
      const errText = await res.text()
      console.error('HF API returned non-OK status:', res.status, errText)
      
      if (res.status === 503 || errText.includes('loading')) {
        return NextResponse.json({ reply: 'The AI model is currently loading into memory. Please try again in about 20 seconds!' })
      }
      return NextResponse.json({ error: 'AI service unavailable' }, { status: 502 })
    }

    const data = await res.json()
    const reply = data.choices?.[0]?.message?.content?.trim()
    console.log(`Success! AI reply: "${reply?.substring(0, 50)}..."`)
    
    return NextResponse.json({ reply: reply || 'No response generated.' })
  } catch (e: any) {
    console.error('Chat API threw an unhandled exception:', e.message, e.stack)
    return NextResponse.json({ error: 'Internal error' }, { status: 500 })
  }
}
