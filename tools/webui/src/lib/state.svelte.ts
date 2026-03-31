import type { AceRequest, AceProps, Song } from './types.js';

const STORAGE_KEY = 'ace';

interface Saved {
	name: string;
	volume: number;
	format: string;
	dark: boolean;
	logsOpen: boolean;
	request: AceRequest;
}

function load(): Saved {
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		if (raw) {
			const parsed = JSON.parse(raw);
			return {
				name: parsed.name || '',
				volume: parsed.volume ?? 0.5,
				format: parsed.format === 'wav' ? 'wav' : 'mp3',
				dark: parsed.dark ?? true,
				logsOpen: parsed.logsOpen ?? true,
				request: parsed.request || { caption: '' }
			};
		}
	} catch {
		// corrupt or unavailable
	}
	return {
		name: '',
		volume: 0.5,
		format: 'mp3',
		dark: false,
		logsOpen: true,
		request: { caption: '' }
	};
}

const saved = load();

export const app = $state({
	name: saved.name,
	volume: saved.volume,
	format: saved.format,
	dark: saved.dark,
	logsOpen: saved.logsOpen,
	request: saved.request as AceRequest,
	songs: [] as Song[],
	props: null as AceProps | null,
	toast: '' as string,
	pendingRequests: [] as AceRequest[],
	pendingIndex: 0,
	refSongId: null as number | null,
	refRangeStart: -1,
	refRangeEnd: -1,
	srcSongId: null as number | null,
	srcRangeStart: -1,
	srcRangeEnd: -1
});

let toastTimer = 0;

export function toast(msg: string, ms = 4000) {
	clearTimeout(toastTimer);
	app.toast = msg;
	toastTimer = setTimeout(() => {
		app.toast = '';
	}, ms) as unknown as number;
}

// overwrite app.request, preserving model routing fields unless the
// incoming request provides them (non-empty string / non-null number).
export function setRequest(incoming: AceRequest) {
	if (!incoming.synth_model) incoming.synth_model = app.request.synth_model;
	if (!incoming.lm_model) incoming.lm_model = app.request.lm_model;
	if (!incoming.lora) incoming.lora = app.request.lora;
	if (incoming.lora_scale == null) incoming.lora_scale = app.request.lora_scale;
	app.request = incoming;
}

// persist on every change
$effect.root(() => {
	$effect(() => {
		const data: Saved = {
			name: app.name,
			volume: app.volume,
			format: app.format,
			dark: app.dark,
			logsOpen: app.logsOpen,
			request: app.request
		};
		localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
	});
});
