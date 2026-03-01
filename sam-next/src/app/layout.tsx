import type { Metadata } from 'next'
import { Plus_Jakarta_Sans } from 'next/font/google'
import './globals.css'
import { AuthProvider } from '@/components/AuthProvider'
import ChatWidget from '@/components/ChatWidget'

const font = Plus_Jakarta_Sans({ subsets: ['latin'], weight: ['300','400','500','600','700','800'], variable: '--font-sans' })

export const metadata: Metadata = {
  title: 'SAM AI — Early Disease Detection',
  description: 'AI-powered early disease detection using machine learning.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${font.variable} antialiased`} style={{ background: '#0a0f1e', color: '#f1f5f9' }}>
        <AuthProvider>
          {children}
          <ChatWidget />
        </AuthProvider>
      </body>
    </html>
  )
}
