// jest-dom adds custom jest matchers for asserting on DOM nodes.
// allows you to do things like:
// expect(element).toHaveTextContent(/react/i)
// learn more: https://github.com/testing-library/jest-dom
import '@testing-library/jest-dom';

class ResizeObserverMock {
	observe() {}
	unobserve() {}
	disconnect() {}
}

global.ResizeObserver = ResizeObserverMock;

HTMLCanvasElement.prototype.getContext = HTMLCanvasElement.prototype.getContext || (() => ({
	fillRect: () => {},
	clearRect: () => {},
	getImageData: () => ({ data: [] }),
	putImageData: () => {},
	createImageData: () => [],
	setTransform: () => {},
	drawImage: () => {},
	save: () => {},
	fillText: () => {},
	restore: () => {},
	beginPath: () => {},
	moveTo: () => {},
	lineTo: () => {},
	closePath: () => {},
	stroke: () => {},
	translate: () => {},
	scale: () => {},
	rotate: () => {},
	arc: () => {},
	fill: () => {},
	measureText: () => ({ width: 0 }),
	transform: () => {},
	rect: () => {},
	clip: () => {},
}));

jest.mock('html2canvas', () => jest.fn(async () => ({
	toDataURL: () => 'data:image/png;base64,TEST',
	width: 100,
	height: 100,
})));

jest.mock('jspdf', () => {
	return function JsPDF() {
		return {
			addImage: jest.fn(),
			addPage: jest.fn(),
			save: jest.fn(),
			internal: {
				pageSize: {
					getWidth: () => 210,
					getHeight: () => 297,
				},
			},
		};
	};
});
