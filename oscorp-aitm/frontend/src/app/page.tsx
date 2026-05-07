import Spline from '@splinetool/react-spline/next';
import ClientOverlay from './ClientOverlay';

export default function Home() {
  return (
    <main className="w-screen h-screen overflow-hidden bg-black flex flex-col relative font-sans">
      <div className="absolute inset-0 z-0 flex items-center justify-center">
        <Spline
          scene="https://prod.spline.design/uvS1H7DMF3caiKfQ/scene.splinecode" 
        />
      </div>
      
      {/* Spline Logo Cover (masks the watermark) */}
      <div className="absolute bottom-0 right-0 w-40 h-16 bg-black z-10 pointer-events-none" />

      <ClientOverlay />
    </main>
  );
}
