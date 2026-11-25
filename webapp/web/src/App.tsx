import { AppProvider } from './AppContext';
import { TopBar } from './components/TopBar';
import { EmptyState } from './components/ui';
import { href, useRoute } from './lib/router';
import { Home } from './pages/Home';
import { Editor } from './pages/editor/Editor';
import { ResultsPage } from './pages/Results';
import { AuralizerPage } from './pages/Auralizer';

export function App() {
  return (
    <AppProvider>
      <Routes />
    </AppProvider>
  );
}

function Routes() {
  const route = useRoute();
  switch (route.name) {
    case 'home':
      return <Home />;
    case 'editor':
      return <Editor key={route.simId} simId={route.simId} />;
    case 'results':
      return <ResultsPage key={route.simId} simId={route.simId} />;
    case 'auralizer':
      return <AuralizerPage key={route.simId} simId={route.simId} />;
    default:
      return (
        <div className="app-shell">
          <TopBar />
          <EmptyState title="Page not found">
            <a href={href.home()}>Back to the home page</a>
          </EmptyState>
        </div>
      );
  }
}
