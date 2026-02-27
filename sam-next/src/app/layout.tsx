import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { AuthProvider } from '@/components/AuthProvider'
import ChatWidget from '@/components/ChatWidget'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'SAM AI — Early Disease Detection',
  description: 'AI-powered early disease detection using machine learning.',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-zinc-50 text-zinc-900 antialiased`}>
        <AuthProvider>
          {children}
          <ChatWidget />
        </AuthProvider>
      </body>
    </html>
  )
}
