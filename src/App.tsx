import Calculator from './components/Calculator';
import { LanguageProvider, useI18n } from './lib/i18n';
import { Languages } from 'lucide-react';

function AppContent() {
  const { t, toggleLang } = useI18n();
  
  return (
    <div className="min-h-screen p-4 md:p-8 lg:py-16 flex justify-center selection:bg-indigo-500/30">
      <div className="w-full max-w-6xl space-y-12">
         <header className="space-y-4 relative">
            <div className="absolute top-0 right-0">
               <button 
                 onClick={toggleLang}
                 className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-zinc-200 dark:bg-zinc-800 text-sm font-medium text-zinc-700 dark:text-zinc-300 hover:bg-zinc-300 dark:hover:bg-zinc-700 transition-colors"
               >
                 <Languages className="w-4 h-4" />
                 {t('langSwitchTo')}
               </button>
            </div>
            <div className="flex items-center gap-3 pr-24">
               <div className="h-10 w-2 bg-indigo-500 rounded-full"></div>
                <h1 className="text-3xl md:text-5xl font-display font-bold tracking-tight text-zinc-900 dark:text-white">
                  KV Cache <span className="text-indigo-500 dark:text-indigo-400">Calculator</span>
                </h1>
            </div>
            <p className="text-zinc-600 dark:text-zinc-400 max-w-2xl text-lg pl-5 border-l border-zinc-300 dark:border-zinc-800">
              {t('subtitle')}
            </p>
         </header>
         
         <Calculator />
      </div>
    </div>
  );
}

export default function App() {
  return (
    <LanguageProvider>
      <AppContent />
    </LanguageProvider>
  );
}

