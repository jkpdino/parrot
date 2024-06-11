'use client'
import Image from 'next/image'
import { Inter } from 'next/font/google'
import styles from './page.module.css'
import React, { useState, useEffect } from 'react'
import { ChatGPTController } from './controller/ChatGPTController'
import ReactMarkdown from 'react-markdown'
import { CSSTransition, TransitionGroup } from 'react-transition-group';
//import {Prism as SyntaxHighlighter} from 'react-syntax-highlighter'
//import {dark} from 'react-syntax-highlighter/dist/esm/styles/prism'


interface IMessage {
    key: number;
    message: string;
    isUser: boolean;
    nodeRef: React.RefObject<HTMLLIElement>;
}

export default function Home() {
    const [messages, setMessages] = useState<IMessage[]>([]);
    const [message, setMessage] = useState('');
  
    const controller = new ChatGPTController();

    function addMessage(message: { message: string; isUser: boolean }) {
        console.log(`adding message ${message.message}`)
        setMessages((messages) => [
          ...messages,
          { ...message, key: messages.length, nodeRef: React.createRef() },
        ]);

        messages.at(messages.length - 1)?.nodeRef.current?.scrollIntoView({ behavior: 'smooth' });
    }

    function addWaitMessage(message: { message: string; isUser: boolean }): number {
        const key = messages.length;

        setMessages((messages) => [
          ...messages,
          { ...message, key, nodeRef: React.createRef() },
        ]);

        messages.at(messages.length - 1)?.nodeRef.current?.scrollIntoView({ behavior: 'smooth' });

        return key;
    }

    function updateMessage(key: number, message: string) {
        const msgs = [...messages];

        console.log(messages.length);

        if (msgs.length > key) {
            console.log("updating");
            msgs[key].message = message;
            msgs[key].nodeRef.current?.scrollIntoView({ behavior: 'smooth' });
        }
        

        setMessages(msgs);
    }

  
    async function sendMessage() {
        //const key = addWaitMessage({ message: "Thinking...", isUser: false });
        const response = await controller.sendMessage(message);
        addMessage({ message: response, isUser: false });
    }

    async function onMessageSend() {
        const newMessage = { message, isUser: true };
        addMessage(newMessage);
        setMessage('');
        await sendMessage();
        
    }
  
    function onMessageChange(event: React.ChangeEvent<HTMLInputElement>) {
      setMessage(event.target.value);
    }
  
    function onKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
      if (event.key === 'Enter') {
        onMessageSend();
      }
    }


    return (
        <main className={styles.main}>
        <div className={styles.description}>
            <p>Parrot</p>
        </div>

        <div className={styles.container}>
            <TransitionGroup component="ul" className={styles.chatContainer}>
                {messages.map((message) => (
                    <CSSTransition key={message.key} timeout={2000} classNames="fade" nodeRef={message.nodeRef}>
                        <li ref={message.nodeRef}><MessageBox message={message.message} isUser={message.isUser} /></li>
                    </CSSTransition>
                    
                ))}
            </TransitionGroup>

            <div className={styles.chatBox}>
            <input
                type="text"
                value={message}
                onKeyDown={onKeyDown}
                onChange={onMessageChange}
                placeholder="Enter prompt..."
            />

            <div className={styles.button} onClick={onMessageSend}>
                <span className="material-symbols-rounded"> send </span>
            </div>
            </div>
        </div>
        </main>
    );
}

function MessageBox(props: { message: string, isUser?: boolean }) {
    const message = props.message;
    const sender = props.isUser ? "Me" : "Parrot";

    const side = props.isUser ? styles.user : styles.bot;

    return (
        <div className={`${styles.message} ${side}`}>
            <p className={styles.callout}>{sender}</p>
            <ReactMarkdown
            /*components={{
                code({node, inline, className, children, ...props}) {
                    const match = /language-(\w+)/.exec(className || '')
                    return !inline && match ? (
                    <SyntaxHighlighter
                        children={String(children).replace(/\n$/, '')}
                        style={dark}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                    />
                    ) : (
                    <code className={className} {...props}>
                        {children}
                    </code>
                    )
                }
                }}*/
                >{message}</ReactMarkdown>
        </div>
    )
}