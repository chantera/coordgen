import React, { useState } from 'react';
import './App.css';

const Selector = () => {
  const [input, setInput] = useState<string>("");
  const [span, setSpan] = useState<[number, number] | null>(null);

  const setText = (v: string) => {
    setInput(v);
    setSpan(null);
  }

  const handleSubmit = (e: any) => {
    e.preventDefault();
    console.log(input);
    console.log(span);
  }

  const handleSelect = (e: any) => {
    const start = e.target.selectionStart;
    const end = e.target.selectionEnd;
    if (end > start) {
      setSpan([start, end]);
    }
  }

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded py-6 px-4 shadow">
      <div className="grid grid-cols-1 sm:grid-cols-4 gap-y-4 gap-x-8">
        <div className="sm:col-span-4">
          <label htmlFor="input" className="block label">Input</label>
          <div className="mt-2">
            <textarea id="input" value={input} onChange={(e) => setText(e.target.value)} onSelect={handleSelect} className="block w-full input" rows={4} />
          </div>
        </div>
        <div className="sm:col-span-2">
          <code className="block h-10 preview truncate">{span && `(${span[0]}, ${span[1]}): ${input.substring(span[0], span[1])}`}</code>
        </div>
        <div>
          <button type="submit" className="block w-full btn btn-primary">Generate</button>
        </div>
        <div>
          <button type="reset" onClick={() => setText("")} className="block w-full btn">Reset</button>
        </div>
      </div>
    </form>
  );
}

const App = () => {
  return (
    <div className="min-h-full">
      <header className="bg-white shadow-sm">
        <div className="mx-auto max-w-4xl py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold tracking-tight text-gray-900">Coordgen</h1>
        </div>
      </header>
      <main>
        <div className="mx-auto max-w-4xl py-6 sm:px-6 lg:px-8">
          <Selector />
        </div>
      </main>
    </div>
  );
}

export default App;
