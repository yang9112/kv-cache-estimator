import { ChangeEvent, useState, useMemo } from 'react';
import { Settings, Cpu, Layers, HardDrive, Database, Zap, AlignLeft, Hash, Info, Type, Server, Sliders } from 'lucide-react';
import { PRESETS, PRECISIONS, KV_CACHE_DTYPES, CalculatorState } from '../types';
import { calculateKV, formatBytes } from '../lib/calc';
import { parseConfigJson } from '../lib/configParser';
import { motion } from 'motion/react';
import { useI18n } from '../lib/i18n';

export default function Calculator() {
  const { t } = useI18n();
  const [state, setState] = useState<CalculatorState>({
    presetId: 'llama-3-8b',
    attentionType: 'standard',
    parameters: 8,
    layers: 32,
    hiddenSize: 4096,
    qHeads: 32,
    kvHeads: 8,
    mlaDc: 512,
    mlaDr: 64,
    fullAttnLayers: 20,
    seqLength: 8192,
    batchSize: 1,
    maxNumBatchedTokens: 8192,
    precision: 2, // Default FP16
    isMoe: false,
    numExperts: 0,
    moeInterSize: 0,
    moeLayers: 0,
    enableExpertParallel: false,
    tp: 1,
    pp: 1,
    gpuMemory: 80,
    gpuUtilization: 0.9,
    dp: 1,
    kvCacheDtype: 'auto',
    blockSize: 16,
    maxModelLen: 8192,
    enforceEager: false,
    enablePrefixCaching: true,
  });

  const groupedPresets = useMemo(() => {
    const groups: Record<string, typeof PRESETS> = {};
    PRESETS.forEach(p => {
      if (!groups[p.family]) groups[p.family] = [];
      groups[p.family].push(p);
    });
    return groups;
  }, []);

  const handlePresetChange = (e: ChangeEvent<HTMLSelectElement>) => {
    const id = e.target.value;
    const preset = PRESETS.find(p => p.id === id);
    if (preset) {
      if (id === 'custom') {
        setState(prev => ({ ...prev, presetId: id }));
      } else {
        setState(prev => ({
          ...prev,
          presetId: id,
          attentionType: preset.attentionType,
          parameters: preset.parameters,
          layers: preset.layers,
          hiddenSize: preset.hiddenSize,
          qHeads: preset.qHeads,
          kvHeads: preset.kvHeads,
          mlaDc: preset.mlaDc ?? prev.mlaDc,
          mlaDr: preset.mlaDr ?? prev.mlaDr,
          fullAttnLayers: preset.fullAttnLayers ?? prev.fullAttnLayers,
          isMoe: preset.isMoe ?? false,
          numExperts: preset.numExperts ?? 0,
          moeInterSize: preset.moeInterSize ?? 0,
          moeLayers: preset.moeLayers ?? 0,
          maxModelLen: prev.maxModelLen,
        }));
      }
    }
  };

  const handleChange = (field: keyof CalculatorState) => (e: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    let value: number | string | boolean = e.target.value;
    
    // Handle checkbox/boolean fields
    if (field === 'enforceEager' || field === 'enablePrefixCaching' || field === 'enableExpertParallel') {
      const checked = (e.target as HTMLInputElement).checked;
      setState(prev => ({ ...prev, [field]: checked }));
      return;
    }
    
    // Handle select (string) fields like kvCacheDtype
    if (field === 'kvCacheDtype' || field === 'presetId' || field === 'attentionType') {
      setState(prev => {
        const newState = { ...prev, [field]: value };
        if (['layers', 'hiddenSize', 'qHeads', 'kvHeads', 'mlaDc', 'mlaDr', 'fullAttnLayers', 'attentionType'].includes(field as string)) {
          newState.presetId = 'custom';
        }
        return newState as CalculatorState;
      });
      return;
    }
    
    // Parse numbers for numeric fields
    if (e.target.type === 'number') {
      value = parseInt(e.target.value, 10);
      if (isNaN(value) || value < 0) value = 0;
    }
    if (field === 'precision' || field === 'gpuUtilization') {
      value = parseFloat(e.target.value);
    }

    setState(prev => {
      const newState = { ...prev, [field]: value };
      // If user manually changes architecture, set preset to 'custom'
      if (['layers', 'hiddenSize', 'qHeads', 'kvHeads', 'mlaDc', 'mlaDr', 'fullAttnLayers', 'attentionType'].includes(field as string)) {
        newState.presetId = 'custom';
      }
      // Auto-set isMoe when numExperts changes
      if (field === 'numExperts') {
        newState.isMoe = (newState.numExperts as number) > 0;
      }
      if (field === 'moeLayers' && (newState.moeLayers as number) > 0 && (newState.numExperts as number) > 0) {
        newState.isMoe = true;
      }
      return newState as CalculatorState;
    });
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const jsonStr = event.target?.result as string;
        const newState = parseConfigJson(jsonStr, state);
        setState(newState);
      } catch (err) {
        alert(t('importError'));
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  };

  const results = useMemo(() => calculateKV(state), [state]);

  const architectureType = useMemo(() => {
    if (state.attentionType === 'mla') return t('mlaAttr');
    if (state.attentionType === 'hybrid') return t('hybridAttr');
    if (state.kvHeads === state.qHeads) return 'MHA';
    if (state.kvHeads === 1) return 'MQA';
    if (state.kvHeads < state.qHeads && state.kvHeads > 1) return 'GQA';
    return t('custom');
  }, [state.kvHeads, state.qHeads, state.attentionType, t]);

  const isMla = state.attentionType === 'mla';
  const isHybrid = state.attentionType === 'hybrid';

  return (
    <>
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
      {/* Left Column: Inputs */}
      <div className="lg:col-span-7 space-y-6">
        
        {/* Model Architecture Panel */}
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="bg-white dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-6 shadow-sm"
        >
          <div className="flex items-center gap-3 mb-6 border-b border-zinc-200 dark:border-zinc-800 pb-4">
            <div className="p-2 bg-indigo-50 dark:bg-indigo-500/10 text-indigo-600 dark:text-indigo-400 rounded-lg">
              <Cpu className="w-5 h-5" />
            </div>
            <h2 className="text-xl font-display font-semibold text-zinc-900 dark:text-white">{t('modelArch')}</h2>
          </div>

          <div className="space-y-5">
            <div className="space-y-2">
              <label className="text-sm font-medium text-zinc-600 dark:text-zinc-400 flex items-center justify-between">
                <span>{t('modelPreset')}</span>
                <div className="flex gap-2 items-center">
                  <span className="text-xs px-2 py-0.5 rounded-full bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300">
                    {architectureType}
                  </span>
                  <label className="text-xs cursor-pointer px-2 py-0.5 rounded-full bg-indigo-50 dark:bg-indigo-500/10 text-indigo-700 dark:text-indigo-400 hover:bg-indigo-100 dark:hover:bg-indigo-500/20 transition-colors shadow-sm">
                    {t('importConfig')}
                    <input type="file" accept=".json" className="hidden" onChange={handleFileChange} />
                  </label>
                </div>
              </label>
              <select 
                value={state.presetId}
                onChange={handlePresetChange}
                className="w-full bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-xl px-4 py-3 text-zinc-900 dark:text-zinc-100 focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all outline-none"
              >
                {Object.entries(groupedPresets).map(([family, presets]) => (
                  <optgroup key={family} label={family === 'Custom' ? t('custom') : family}>
                    {presets.map(p => (
                      <option key={p.id} value={p.id}>{p.id === 'custom' ? t('custom') : p.name}</option>
                    ))}
                  </optgroup>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="sm:col-span-2">
                 <label className="text-sm font-medium text-zinc-600 dark:text-zinc-400 flex items-center gap-2 mb-2">
                    <Type className="w-4 h-4" /> {t('attentionType')}
                 </label>
                 <div className="flex gap-4">
                    <label className="flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-300 cursor-pointer">
                       <input 
                         type="radio" 
                         name="attentionType" 
                         value="standard" 
                         checked={state.attentionType === 'standard'} 
                         onChange={handleChange('attentionType')}
                         className="text-indigo-500 focus:ring-indigo-500"
                       />
                       {t('standardAttr')}
                    </label>
                    <label className="flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-300 cursor-pointer">
                       <input 
                         type="radio" 
                         name="attentionType" 
                         value="mla" 
                         checked={isMla} 
                         onChange={handleChange('attentionType')}
                         className="text-indigo-500 focus:ring-indigo-500"
                       />
                       {t('mlaAttr')}
                    </label>
                    <label className="flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-300 cursor-pointer">
                       <input 
                         type="radio" 
                         name="attentionType" 
                         value="hybrid" 
                         checked={isHybrid} 
                         onChange={handleChange('attentionType')}
                         className="text-indigo-500 focus:ring-indigo-500"
                       />
                       {t('hybridAttr')}
                    </label>
                 </div>
              </div>

              <InputGroup 
                label={t('layers')} 
                icon={<Layers className="w-4 h-4" />}
                value={state.layers}
                onChange={handleChange('layers')}
              />
              
              {!isMla && (
                <>
                  {isHybrid && (
                    <InputGroup 
                      label={t('fullAttnLayers')} 
                      icon={<Layers className="w-4 h-4" />}
                      value={state.fullAttnLayers}
                      onChange={handleChange('fullAttnLayers')}
                      helpText="Layers using full KV cache"
                    />
                  )}
                  <InputGroup 
                    label={t('hiddenSize')} 
                    icon={<Settings className="w-4 h-4" />}
                    value={state.hiddenSize}
                    onChange={handleChange('hiddenSize')}
                    step={256}
                  />
                  <InputGroup 
                    label={t('qHeads')} 
                    icon={<Hash className="w-4 h-4" />}
                    value={state.qHeads}
                    onChange={handleChange('qHeads')}
                  />
                  <InputGroup 
                    label={t('kvHeads')} 
                    icon={<Hash className="w-4 h-4" />}
                    value={state.kvHeads}
                    onChange={handleChange('kvHeads')}
                    helpText={t('kvHeadsHelp')}
                  />
                </>
              )}

              {isMla && (
                <>
                  <InputGroup 
                    label={t('mlaDc')}
                    icon={<Hash className="w-4 h-4" />}
                    value={state.mlaDc}
                    onChange={handleChange('mlaDc')}
                  />
                  <InputGroup 
                    label={t('mlaDr')}
                    icon={<Hash className="w-4 h-4" />}
                    value={state.mlaDr}
                    onChange={handleChange('mlaDr')}
                  />
                </>
              )}

              {/* MoE (Mixture of Experts) Fields — shown when isMoe or when numExperts > 0 */}
              {(state.isMoe || state.numExperts > 0) && (
                <>
                  <hr className="sm:col-span-2 border-zinc-200 dark:border-zinc-800 my-2" />
                  <div className="sm:col-span-2">
                    <p className="text-xs font-medium text-pink-600 dark:text-pink-400 mb-2">
                      MoE {t('isMoe')} — {t('numExperts')}: {state.numExperts || '–'}
                    </p>
                  </div>
                  <InputGroup 
                    label={t('numExperts')} 
                    icon={<Hash className="w-4 h-4" />}
                    value={state.numExperts}
                    onChange={handleChange('numExperts')}
                  />
                  <InputGroup 
                    label={t('moeInterSize')} 
                    icon={<Hash className="w-4 h-4" />}
                    value={state.moeInterSize}
                    onChange={handleChange('moeInterSize')}
                  />
                  <InputGroup 
                    label={t('moeLayers')} 
                    icon={<Layers className="w-4 h-4" />}
                    value={state.moeLayers}
                    onChange={handleChange('moeLayers')}
                  />
                </>
              )}
            </div>
          </div>
        </motion.div>

        {/* Inference Config Panel */}
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.1 }}
          className="bg-white dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-6 shadow-sm"
        >
          <div className="flex items-center gap-3 mb-6 border-b border-zinc-200 dark:border-zinc-800 pb-4">
            <div className="p-2 bg-emerald-50 dark:bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 rounded-lg">
              <Zap className="w-5 h-5" />
            </div>
            <h2 className="text-xl font-display font-semibold text-zinc-900 dark:text-white">{t('inferenceConfig')}</h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
             <div className="sm:col-span-2 space-y-2">
              <label className="text-sm font-medium text-zinc-600 dark:text-zinc-400 flex items-center gap-2">
                <Database className="w-4 h-4" /> {t('dataType')}
              </label>
              <select 
                value={state.precision}
                onChange={handleChange('precision')}
                className="w-full bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-xl px-4 py-3 text-zinc-900 dark:text-zinc-100 focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500 transition-all outline-none"
              >
                {PRECISIONS.map(p => (
                  <option key={p.bytes} value={p.bytes}>{p.label}</option>
                ))}
              </select>
            </div>
            
            <InputGroup 
              label={t('seqLength')} 
              icon={<AlignLeft className="w-4 h-4" />}
              value={state.seqLength}
              onChange={handleChange('seqLength')}
              step={1024}
              helpText={t('seqLengthHelp')}
            />
            <InputGroup 
              label={t('batchSize')} 
              icon={<Layers className="w-4 h-4" />}
              value={state.batchSize}
              onChange={handleChange('batchSize')}
            />
            <InputGroup 
              label={t('maxNumBatchedTokens')} 
              icon={<Layers className="w-4 h-4" />}
              value={state.maxNumBatchedTokens}
              onChange={handleChange('maxNumBatchedTokens')}
              step={1024}
              helpText={t('maxNumBatchedTokensHelp')}
            />
          </div>
        </motion.div>

        {/* Deployment & Parallelism Config Panel */}
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.2 }}
          className="bg-white dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-6 shadow-sm"
        >
          <div className="flex items-center gap-3 mb-6 border-b border-zinc-200 dark:border-zinc-800 pb-4">
            <div className="p-2 bg-pink-50 dark:bg-pink-500/10 text-pink-600 dark:text-pink-400 rounded-lg">
              <Server className="w-5 h-5" />
            </div>
            <h2 className="text-xl font-display font-semibold text-zinc-900 dark:text-white">{t('deploymentConfig')}</h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            <InputGroup 
              label={t('parameters')} 
              icon={<Hash className="w-4 h-4" />}
              value={state.parameters}
              onChange={handleChange('parameters')}
              step={1}
            />
            <div className="hidden sm:block"></div>
            <InputGroup 
              label={t('tensorParallel')} 
              icon={<Cpu className="w-4 h-4" />}
              value={state.tp}
              onChange={handleChange('tp')}
            />
            <InputGroup 
              label={t('pipelineParallel')} 
              icon={<Server className="w-4 h-4" />}
              value={state.pp}
              onChange={handleChange('pp')}
            />
            <InputGroup 
              label={t('dataParallel')} 
              icon={<Server className="w-4 h-4" />}
              value={state.dp}
              onChange={handleChange('dp')}
              helpText={t('dpHelp')}
            />
            <hr className="sm:col-span-2 border-zinc-200 dark:border-zinc-800 my-2" />
            <div className="sm:col-span-2 space-y-2">
              <label className="text-sm font-medium text-zinc-600 dark:text-zinc-400 flex items-center gap-2">
                <Cpu className="w-4 h-4" /> {t('enableExpertParallel')}
              </label>
              <label className="flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-300 cursor-pointer py-3">
                <input 
                  type="checkbox" 
                  checked={state.enableExpertParallel} 
                  onChange={handleChange('enableExpertParallel')}
                  className="text-pink-500 focus:ring-pink-500 w-4 h-4"
                />
                {t('epHelp')}
              </label>
            </div>
            {state.isMoe && (
              <>
                <div className="sm:col-span-2 space-y-1">
                  <p className="text-xs text-zinc-500 font-mono">
                    {t('expertParams')}: {results.expertParams > 0 ? (results.expertParams / 1e9).toFixed(1) + 'B' : '–'}
                    {' · '}{t('denseParams')}: {results.expertParams > 0 ? ((state.parameters || 0) - results.expertParams / 1e9).toFixed(1) + 'B' : (state.parameters || 0) + 'B'}
                    {state.enableExpertParallel && results.epSize > 1 && (
                      <> {' · '}{t('epSize')}: TP×DP={results.epSize} {' · '}{t('localExperts')}: ~{results.localNumExperts}</>
                    )}
                  </p>
                </div>
              </>
            )}
            <hr className="sm:col-span-2 border-zinc-200 dark:border-zinc-800 my-2" />
            <InputGroup 
              label={t('gpuMemory')} 
              icon={<HardDrive className="w-4 h-4" />}
              value={state.gpuMemory}
              onChange={handleChange('gpuMemory')}
            />
            <InputGroup 
              label={t('gpuUtilization')} 
              icon={<Zap className="w-4 h-4" />}
              value={state.gpuUtilization}
              onChange={handleChange('gpuUtilization')}
              step={0.05}
            />
          </div>
        </motion.div>

        {/* vLLM Advanced Config Panel */}
        <motion.div 
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3, delay: 0.25 }}
          className="bg-white dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-6 shadow-sm"
        >
          <div className="flex items-center gap-3 mb-6 border-b border-zinc-200 dark:border-zinc-800 pb-4">
            <div className="p-2 bg-amber-50 dark:bg-amber-500/10 text-amber-600 dark:text-amber-400 rounded-lg">
              <Sliders className="w-5 h-5" />
            </div>
            <h2 className="text-xl font-display font-semibold text-zinc-900 dark:text-white">{t('vllmConfig')}</h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            <div className="sm:col-span-2 space-y-2">
              <label className="text-sm font-medium text-zinc-600 dark:text-zinc-400 flex items-center gap-2">
                <Database className="w-4 h-4" /> {t('kvCacheDtype')}
              </label>
              <select 
                value={state.kvCacheDtype}
                onChange={handleChange('kvCacheDtype')}
                className="w-full bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-xl px-4 py-3 text-zinc-900 dark:text-zinc-100 focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500 transition-all outline-none"
              >
                {KV_CACHE_DTYPES.map(d => (
                  <option key={d.key} value={d.key}>{d.label}</option>
                ))}
              </select>
              <p className="text-xs text-zinc-500">{t('kvCacheDtypeHelp')}</p>
            </div>

            <InputGroup 
              label={t('blockSize')} 
              icon={<Layers className="w-4 h-4" />}
              value={state.blockSize}
              onChange={handleChange('blockSize')}
              helpText={t('blockSizeHelp')}
            />
            <InputGroup 
              label={t('maxModelLen')} 
              icon={<AlignLeft className="w-4 h-4" />}
              value={state.maxModelLen}
              onChange={handleChange('maxModelLen')}
              step={1024}
              helpText={t('maxModelLenHelp')}
            />

            <div className="space-y-2">
              <label className="text-sm font-medium text-zinc-600 dark:text-zinc-400 flex items-center gap-2">
                <Zap className="w-4 h-4" /> {t('enforceEager')}
              </label>
              <label className="flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-300 cursor-pointer py-3">
                <input 
                  type="checkbox" 
                  checked={state.enforceEager} 
                  onChange={handleChange('enforceEager')}
                  className="text-amber-500 focus:ring-amber-500 w-4 h-4"
                />
                {t('enforceEagerHelp')}
              </label>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium text-zinc-600 dark:text-zinc-400 flex items-center gap-2">
                <Database className="w-4 h-4" /> {t('enablePrefixCaching')}
              </label>
              <label className="flex items-center gap-2 text-sm text-zinc-700 dark:text-zinc-300 cursor-pointer py-3">
                <input 
                  type="checkbox" 
                  checked={state.enablePrefixCaching} 
                  onChange={handleChange('enablePrefixCaching')}
                  className="text-amber-500 focus:ring-amber-500 w-4 h-4"
                />
                {t('enablePrefixCachingHelp')}
              </label>
            </div>
          </div>
        </motion.div>

      </div>

      {/* Right Column: Results */}
      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.4 }}
        className="lg:col-span-5 sticky top-8 space-y-6"
      >
        <div className="bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-500/10 dark:to-purple-500/5 border border-indigo-200 dark:border-indigo-500/20 rounded-3xl p-8 relative overflow-hidden backdrop-blur-sm shadow-sm">
           {/* Decorative background glow */}
           <div className="absolute top-0 right-0 -mr-16 -mt-16 w-64 h-64 bg-indigo-500 max-w-full rounded-full mix-blend-screen filter blur-[100px] opacity-10 dark:opacity-20 pointer-events-none"></div>

           <div className="relative z-10">
              <div className="flex items-center gap-3 text-indigo-600 dark:text-indigo-300 mb-8">
                <HardDrive className="w-6 h-6" />
                <h3 className="text-lg font-medium tracking-wide">{t('vramPerGPU')}</h3>
              </div>

              <div className="space-y-1 mb-10 text-center md:text-left border-b border-indigo-200 dark:border-indigo-500/10 pb-6">
                 <div className="text-5xl lg:text-7xl font-display font-bold text-zinc-900 dark:text-white tracking-tighter">
                   {formatBytes(results.totalPerGPU).split(' ')[0]}
                   <span className="text-2xl lg:text-4xl text-indigo-500 dark:text-indigo-400 ml-2 font-medium">
                     {formatBytes(results.totalPerGPU).split(' ')[1]}
                   </span>
                 </div>
                 <p className="text-zinc-500 dark:text-zinc-400 font-mono text-sm mt-3 h-5">
                 </p>
              </div>

              <div className="space-y-4 pt-4 flex-1 flex flex-col justify-end">
                 <ResultRow 
                   label={t('kvPerGPU')}
                   value={formatBytes(results.kvPerGPU)}
                   highlight
                 />
                 <ResultRow 
                   label={t('weightPerGPU')}
                   value={formatBytes(results.weightPerGPU) + (state.isMoe && results.expertWeightTotal > 0 ? ' (D:' + formatBytes(results.denseWeightPerGPU).split(' ')[0] + ' E:' + formatBytes(results.expertWeightPerGPU).split(' ')[0] + ')' : '')}
                 />
                 <ResultRow 
                   label={t('overheadVram')}
                   value={formatBytes(results.overheadPerGPU)}
                 />
              </div>
           </div>
        </div>

        {/* vLLM Metrics */}
        <div className="bg-white dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-6 shadow-sm">
          <div className="flex items-center gap-2 mb-4">
            <h3 className="text-sm font-medium text-zinc-800 dark:text-zinc-200">{t('vllmBudgetHeading')}</h3>
          </div>
          <p className="text-xs text-zinc-500 mb-4">{t('vllmHelpText')}</p>
          <div className="space-y-3">
             <ResultRow 
               label={t('maxUsableKvBudget')}
               value={formatBytes(results.vllmKvBudgetPerGPU)}
               highlight
             />
             <ResultRow 
               label={t('maxTokensPerGpu')}
               value={results.maxTokensPerGPU > 0 ? Math.floor(results.maxTokensPerGPU).toLocaleString() : 'Out of Memory'}
               highlight
             />
          </div>
        </div>

        <div className="bg-white dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-6 shadow-sm">
          <h3 className="text-sm font-medium text-zinc-800 dark:text-zinc-200 mb-4">{t('vllmBlockMetrics')}</h3>
          <div className="space-y-3">
             <ResultRow 
               label={t('numBlocks')}
               value={results.numBlocks.toLocaleString()}
             />
             <ResultRow 
               label={t('kvCacheTokens')}
               value={results.kvCacheTokensPerGPU.toLocaleString()}
               highlight
             />
             <ResultRow 
               label={t('maxConcurrency')}
               value={results.maxConcurrency > 0 ? results.maxConcurrency.toFixed(2) + 'x' : 'Out of Memory'}
               highlight
             />
             <hr className="my-2 border-zinc-200 dark:border-zinc-800" />
             <ResultRow 
               label={t('totalGpus')}
               value={results.totalGpus.toLocaleString() + ' (TP=' + state.tp + ' PP=' + state.pp + ' DP=' + state.dp + ')' + (state.enableExpertParallel ? ' EP=on' : '')}
             />
             <ResultRow 
               label={t('totalClusterKVTokens')}
               value={results.totalClusterKVTokens.toLocaleString()}
               highlight
             />
          </div>
        </div>

        <div className="bg-white dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-6 shadow-sm">
          <h3 className="text-sm font-medium text-zinc-800 dark:text-zinc-200 mb-4">{t('totalMemory')}</h3>
          <div className="space-y-3">
            <ResultRow 
               label="Total KV Cache"
               value={formatBytes(results.totalMemory)} 
             />
             <ResultRow 
               label={t('weightMemory')}
               value={formatBytes(results.weightTotal)} 
             />
             <ResultRow 
               label={t('tokensPerGb')}
               value={Math.floor(results.tokensPerGB).toLocaleString()}
             />
             <hr className="my-2 border-zinc-200 dark:border-zinc-800" />
             <ResultRow 
               label={t('headDim')}
               value={isMla ? t('notApplicable') : results.headDim.toString()} 
             />
             <ResultRow 
               label={t('bytesPerToken1L')}
               value={formatBytes(results.sizePerTokenPerLayer)}
             />
             <ResultRow 
               label={t('bytesPerTokenAll')}
               value={formatBytes(results.sizePerTokenTotal)} 
             />
          </div>
        </div>
        
        {/* Formula Hint */}
        <div className="mt-6 px-6 py-4 rounded-xl bg-white dark:bg-zinc-900/40 border border-zinc-200 dark:border-zinc-800 text-sm text-zinc-600 dark:text-zinc-500 flex items-start gap-3 shadow-sm">
           <Info className="w-5 h-5 text-zinc-400 shrink-0 mt-0.5" />
           <p className="leading-relaxed">
             <strong className="text-zinc-700 dark:text-zinc-300">{t('formula')}:</strong> <br/>
             <span className="font-mono text-xs break-all">
                {isMla ? t('formulaMla') : isHybrid ? t('formulaHybrid') : t('formulaStandard')}
             </span>
           </p>
        </div>
      </motion.div>
    </div>

    {/* Educational Context Panel */}
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, delay: 0.4 }}
      className="mt-8 bg-zinc-50 dark:bg-zinc-900/50 border border-zinc-200 dark:border-zinc-800 rounded-2xl p-6 shadow-sm"
    >
      <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100 mb-4">{t('explanationTitle')}</h3>
      <div className="prose prose-sm dark:prose-invert prose-indigo max-w-none text-zinc-600 dark:text-zinc-400 space-y-4">
        <p><strong>{t('formulaDesc1')}</strong></p>
        <p>{t('formulaDesc2')}</p>
        <ul className="list-disc pl-5 space-y-2">
          <li><strong>{t('expWeight')}</strong> {t('expWeightDesc')}</li>
          <li><strong>{t('expKV')}</strong> {t('expKVDesc')}</li>
          <li><strong>{t('expAct')}</strong> {t('expActDesc')}</li>
          <li><strong>{t('expTpPp')}</strong> {t('expTpPpDesc')}</li>
          <li><strong>{t('expDp')}</strong> {t('expDpDesc')}</li>
          <li><strong>{t('expEp')}</strong> {t('expEpDesc')}</li>
        </ul>
        <p className="pt-2 italic text-xs text-zinc-500">{t('expEnd')}</p>
      </div>
    </motion.div>
    </>
  );
}

// Subcomponents
function InputGroup({ 
  label, 
  value, 
  onChange, 
  type = "number", 
  icon,
  step = 1,
  helpText
}: { 
  label: string, 
  value: number | string, 
  onChange: (e: ChangeEvent<HTMLInputElement>) => void,
  type?: string,
  icon?: React.ReactNode,
  step?: number,
  helpText?: string
}) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-zinc-600 dark:text-zinc-400 flex items-center justify-between">
        <div className="flex items-center gap-2">
           {icon} {label}
        </div>
      </label>
      <input 
        type={type}
        value={value}
        onChange={onChange}
        step={step}
        className="w-full bg-zinc-50 dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 rounded-xl px-4 py-3 text-zinc-900 dark:text-zinc-100 font-mono focus:ring-2 focus:ring-indigo-500/50 focus:border-indigo-500 transition-all outline-none"
      />
      {helpText && <p className="text-xs text-zinc-500">{helpText}</p>}
    </div>
  );
}

function ResultRow({ label, value, highlight = false }: { label: string, value: string, highlight?: boolean }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-zinc-600 dark:text-zinc-400 text-sm">{label}</span>
      <span className={`font-mono font-medium ${highlight ? 'text-indigo-700 dark:text-indigo-300 bg-indigo-100 dark:bg-indigo-500/10 px-3 py-1 rounded-md' : 'text-zinc-800 dark:text-zinc-200'}`}>
        {value}
      </span>
    </div>
  );
}
