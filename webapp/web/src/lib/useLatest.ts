import { useCallback, useLayoutEffect, useRef } from 'react';

/**
 * Stable callback that always runs the latest closure. Used for handlers passed
 * into the R3F canvas, whose subtree may receive new props a little later than
 * the DOM: a click right after a tab switch must see the new state.
 */
export function useLatest<A extends unknown[], R>(fn: (...args: A) => R): (...args: A) => R {
  const ref = useRef(fn);
  ref.current = fn;
  useLayoutEffect(() => {
    ref.current = fn;
  });
  return useCallback((...args: A) => ref.current(...args), []);
}
