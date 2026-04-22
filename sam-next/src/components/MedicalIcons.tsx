import {
  HeartPulse, FlaskConical, Activity, Syringe, Microscope,
  Stethoscope, Brain, Sparkles, Zap, ShieldCheck, TrendingUp,
  FileText, Info, ChevronDown, BarChart3, Award, Database,
  Lock, Globe, CheckCircle, AlertTriangle, Calendar, BookOpen,
  User, Cpu, Server, Code2, Layers, Target, Clock, Eye,
  ArrowRight, Plus, Minus, X, Check, Search, Filter, Download,
  Printer, RefreshCw, ChevronRight, ChevronUp, LayoutDashboard,
  LogOut, Menu, Gauge, Beaker, Dna, ScanLine,
} from 'lucide-react'
import type { CSSProperties } from 'react'

interface IconProps { size?: number; color?: string; strokeWidth?: number; style?: CSSProperties }

// ── Diagnostic module icons ──────────────────────────────────
export function HeartECGIcon(p: IconProps) { return <HeartPulse size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function LiverIcon(p: IconProps)    { return <FlaskConical size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function GlucoseDropIcon(p: IconProps) { return <Gauge size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function SyringeIcon(p: IconProps)  { return <Syringe size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function MicroscopeIcon(p: IconProps) { return <Microscope size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function DnaIcon(p: IconProps)      { return <Dna size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function ScanIcon(p: IconProps)     { return <ScanLine size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }

// ── General UI icons ─────────────────────────────────────────
export function StethoscopeIcon(p: IconProps)    { return <Stethoscope size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function BrainSparkIcon(p: IconProps)     { return <Brain size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function SparklesIcon(p: IconProps)       { return <Sparkles size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function ActivityIcon(p: IconProps)       { return <Activity size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function ZapIcon(p: IconProps)            { return <Zap size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function ShieldCheckIcon(p: IconProps)    { return <ShieldCheck size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function TrendingUpIcon(p: IconProps)     { return <TrendingUp size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function FileTextIcon(p: IconProps)       { return <FileText size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function InfoIcon(p: IconProps)           { return <Info size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function ChevronDownIcon(p: IconProps)    { return <ChevronDown size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function ChevronRightIcon(p: IconProps)   { return <ChevronRight size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function ChevronUpIcon(p: IconProps)      { return <ChevronUp size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function BarChart3Icon(p: IconProps)      { return <BarChart3 size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function AwardIcon(p: IconProps)          { return <Award size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function DatabaseIcon(p: IconProps)       { return <Database size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function LockIcon(p: IconProps)           { return <Lock size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function GlobeIcon(p: IconProps)          { return <Globe size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function CheckCircleIcon(p: IconProps)    { return <CheckCircle size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function AlertTriangleIcon(p: IconProps)  { return <AlertTriangle size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function CalendarIcon(p: IconProps)       { return <Calendar size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function BookOpenIcon(p: IconProps)       { return <BookOpen size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function UserIcon(p: IconProps)           { return <User size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function CpuIcon(p: IconProps)            { return <Cpu size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function ServerIcon(p: IconProps)         { return <Server size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function Code2Icon(p: IconProps)          { return <Code2 size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function LayersIcon(p: IconProps)         { return <Layers size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function TargetIcon(p: IconProps)         { return <Target size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function ClockIcon(p: IconProps)          { return <Clock size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function EyeIcon(p: IconProps)            { return <Eye size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function ArrowRightIcon(p: IconProps)     { return <ArrowRight size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function PlusIcon(p: IconProps)           { return <Plus size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function MinusIcon(p: IconProps)          { return <Minus size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function XIcon(p: IconProps)              { return <X size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function CheckIcon(p: IconProps)          { return <Check size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function SearchIcon(p: IconProps)         { return <Search size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function FilterIcon(p: IconProps)         { return <Filter size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function DownloadIcon(p: IconProps)       { return <Download size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function PrinterIcon(p: IconProps)        { return <Printer size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function RefreshIcon(p: IconProps)        { return <RefreshCw size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function DashboardIcon(p: IconProps)      { return <LayoutDashboard size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function LogOutIcon(p: IconProps)         { return <LogOut size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function MenuIcon(p: IconProps)           { return <Menu size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
export function BeakerIcon(p: IconProps)         { return <Beaker size={p.size??24} color={p.color??'currentColor'} strokeWidth={p.strokeWidth??1.6} style={p.style} /> }
