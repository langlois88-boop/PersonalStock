import { render, screen } from '@testing-library/react';
import App from './App';

test('renders dashboard shell', () => {
  render(<App />);
  expect(screen.getByText(/Dashboard Home/i)).toBeInTheDocument();
  expect(screen.getByText(/Command Center/i)).toBeInTheDocument();
});
