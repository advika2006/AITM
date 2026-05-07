"use client";

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Activity, BrainCircuit, Cpu, ArrowRight, X } from 'lucide-react';

export default function ClientOverlay() {
  const [showApp, setShowApp] = useState(false);

  const features = [
    {
      icon: <Activity className="w-5 h-5 text-blue-400" />,
      title: "Real-time Lip Tracking",
      description: "Utilizes computer vision to accurately detect and track lip movements in real-time from camera feed.",
    },
    {
      icon: <BrainCircuit className="w-5 h-5 text-emerald-400" />,
      title: "Silent Speech to Text",
      description: "Converts silent lip movements into readable text without any audio input using deep learning models.",
    },
    {
      icon: <Cpu className="w-5 h-5 text-purple-400" />,
      title: "AI-Powered Accessibility",
      description: "Provides communication assistance for noisy environments and people with speech/hearing disabilities through gesture recognition and multi-language support.",
    },
  ];

  return (
    <>
      <div className="absolute inset-0 z-10 w-full h-full flex flex-col justify-end items-center pb-12 px-6 pointer-events-none">
        
        {/* Action Button */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="mb-8 pointer-events-auto"
        >
          <button 
            onClick={() => setShowApp(true)}
            className="px-5 py-2.5 rounded-xl bg-[#111111]/30 backdrop-blur-md hover:bg-[#111111]/50 border border-white/20 transition-all font-medium text-xs flex items-center gap-2 text-gray-200 hover:text-white hover:shadow-[0_0_15px_rgba(255,255,255,0.1)]"
          >
            Launch Platform <ArrowRight className="w-3 h-3" />
          </button>
        </motion.div>

        {/* Feature Cards Grid - Anchored to the bottom */}
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.4 }}
          className="grid md:grid-cols-3 gap-6 w-full max-w-5xl pointer-events-auto"
        >
          {features.map((feature, idx) => (
            <div
              key={idx}
              className="bg-transparent backdrop-blur-sm border border-white/5 p-5 rounded-2xl flex flex-col gap-2 hover:bg-white/5 hover:border-white/20 transition-all text-white"
            >
              <div className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center border border-white/10">
                {feature.icon}
              </div>
              <h3 className="text-sm font-semibold text-gray-100">{feature.title}</h3>
              <p className="text-[11px] text-gray-400 leading-relaxed font-light">
                {feature.description}
              </p>
            </div>
          ))}
        </motion.div>
      </div>

      {/* Streamlit App Overlay */}
      <AnimatePresence>
        {showApp && (
          <motion.div 
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className="fixed inset-0 z-50 bg-black/60 backdrop-blur-2xl flex flex-col"
          >
            {/* Top Bar / Close Button */}
            <div className="w-full h-12 bg-black/40 border-b border-white/10 flex items-center justify-between px-6 shrink-0 backdrop-blur-md">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
                <span className="text-xs font-medium text-gray-300">Oscorp AITM System Active</span>
              </div>
              <button 
                onClick={() => setShowApp(false)}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-red-500/20 hover:text-red-400 text-gray-400 transition-colors text-xs font-medium border border-white/10"
              >
                <X className="w-3 h-3" /> Close App
              </button>
            </div>
            
            {/* Embedded Streamlit */}
            <iframe 
              src="http://localhost:8501" 
              className="w-full flex-1 border-none bg-transparent"
              title="Streamlit Backend"
            />
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
