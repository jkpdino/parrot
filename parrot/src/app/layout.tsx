import './globals.css'
import './animate.css'

export const metadata = {
  title: 'Parrot',
  description: 'Your helpful AI assistant',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@48,400,1,0" />
      </head>
      <body>{children}</body>
    </html>
  )
}
