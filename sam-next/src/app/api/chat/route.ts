import { NextRequest, NextResponse } from 'next/server'
import { HfInference } from '@huggingface/inference'

export async function POST(req: NextRequest) {
  const { message } = await req.json()
  if (!message?.trim()) return NextResponse.json({ error: 'Empty message' }, { status: 400 })

  const token = process.env.HF_TOKEN
  if (!token) return NextResponse.json({ error: 'Server misconfiguration' }, { status: 500 })

  const hf = new HfInference(token)
  
  try {
    const response = await hf.chatCompletion({
      model: 'meta-llama/Llama-3.2-3B-Instruct',
      messages: [
        { role: 'system', content: 'You are SAM, a helpful medical AI assistant. Answer concisely and clearly in a friendly tone. Do not provide medical diagnoses.' },
        { role: 'user', content: message }
      ],
      max_tokens: 300,
      temperature: 0.4,
    })

    const reply = response.choices[0]?.message?.content?.trim()
    return NextResponse.json({ reply: reply || 'No response generated.' })
  } catch (e: any) {
    console.error('Chat route error:', e.message || e)
    
    // Fallback to a different free model if Mistral is loading/unavailable
    if (e.message?.includes('loading') || e.message?.includes('503')) {
       return NextResponse.json({ reply: 'The AI model is currently loading into memory. Please try again in about 20 seconds!' })
    }
    
    return NextResponse.json({ error: 'AI service unavailable' }, { status: 502 })
  }
}
