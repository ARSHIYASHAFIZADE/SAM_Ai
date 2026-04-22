import { NextRequest, NextResponse } from 'next/server'

export async function POST(req: NextRequest) {
  try {
    const { message } = await req.json()
    if (!message?.trim()) {
      return NextResponse.json({ error: 'Empty message' }, { status: 400 })
    }

    const key = process.env.GROQ_API_KEY
    if (!key) {
      console.error('Chat error: missing GROQ_API_KEY')
      return NextResponse.json({ error: 'Server misconfiguration' }, { status: 500 })
    }

    const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${key}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'llama-3.3-70b-versatile',
        messages: [
          {
            role: 'system',
            content: 'You are SAM, a helpful clinical AI assistant built into a medical diagnostic platform. Answer concisely and clearly in a friendly, professional tone. Help users understand health conditions, risk factors, lab values, and how to interpret their diagnostic results. Do not provide definitive medical diagnoses — always recommend consulting a qualified healthcare professional for medical decisions. Keep responses under 200 words.',
          },
          { role: 'user', content: message },
        ],
        max_tokens: 350,
        temperature: 0.4,
      }),
    })

    if (!res.ok) {
      const errText = await res.text()
      console.error('Groq error:', res.status, errText)
      return NextResponse.json({ error: 'AI service unavailable' }, { status: 502 })
    }

    const data = await res.json()
    const reply = data.choices?.[0]?.message?.content?.trim()
    return NextResponse.json({ reply: reply || 'No response generated.' })
  } catch (e: unknown) {
    console.error('Chat API error:', e)
    return NextResponse.json({ error: 'Internal error' }, { status: 500 })
  }
}
