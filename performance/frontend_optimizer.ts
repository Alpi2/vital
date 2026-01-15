/**
 * Frontend Rendering Optimization
 * Optimizes Angular application performance
 */

import { ChangeDetectionStrategy, ChangeDetectorRef } from '@angular/core';

/**
 * Virtual scrolling for large lists
 */
export class VirtualScrollOptimizer {
  private itemHeight: number;
  private containerHeight: number;
  private scrollTop: number = 0;

  constructor(itemHeight: number, containerHeight: number) {
    this.itemHeight = itemHeight;
    this.containerHeight = containerHeight;
  }

  /**
   * Calculate visible range for virtual scrolling
   */
  getVisibleRange(totalItems: number): { start: number; end: number } {
    const visibleItems = Math.ceil(this.containerHeight / this.itemHeight);
    const start = Math.floor(this.scrollTop / this.itemHeight);
    const end = Math.min(start + visibleItems + 2, totalItems); // +2 for buffer

    return { start, end };
  }

  updateScrollPosition(scrollTop: number): void {
    this.scrollTop = scrollTop;
  }

  getTotalHeight(totalItems: number): number {
    return totalItems * this.itemHeight;
  }

  getOffsetY(start: number): number {
    return start * this.itemHeight;
  }
}

/**
 * Debounce decorator for expensive operations
 */
export function Debounce(delay: number = 300) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const original = descriptor.value;
    let timeout: any;

    descriptor.value = function (...args: any[]) {
      clearTimeout(timeout);
      timeout = setTimeout(() => {
        original.apply(this, args);
      }, delay);
    };

    return descriptor;
  };
}

/**
 * Throttle decorator for high-frequency events
 */
export function Throttle(delay: number = 100) {
  return function (target: any, propertyKey: string, descriptor: PropertyDescriptor) {
    const original = descriptor.value;
    let lastCall = 0;

    descriptor.value = function (...args: any[]) {
      const now = Date.now();
      if (now - lastCall >= delay) {
        lastCall = now;
        original.apply(this, args);
      }
    };

    return descriptor;
  };
}

/**
 * Memoization for expensive computations
 */
export class MemoizationCache<T> {
  private cache = new Map<string, { value: T; timestamp: number }>();
  private ttl: number;

  constructor(ttlMs: number = 60000) {
    this.ttl = ttlMs;
  }

  get(key: string): T | undefined {
    const cached = this.cache.get(key);
    if (!cached) return undefined;

    const now = Date.now();
    if (now - cached.timestamp > this.ttl) {
      this.cache.delete(key);
      return undefined;
    }

    return cached.value;
  }

  set(key: string, value: T): void {
    this.cache.set(key, {
      value,
      timestamp: Date.now(),
    });
  }

  clear(): void {
    this.cache.clear();
  }

  size(): number {
    return this.cache.size;
  }
}

/**
 * Lazy loading helper
 */
export class LazyLoader {
  private loadedModules = new Set<string>();

  async loadModule(moduleName: string, loader: () => Promise<any>): Promise<any> {
    if (this.loadedModules.has(moduleName)) {
      console.log(`Module ${moduleName} already loaded`);
      return;
    }

    console.log(`Loading module ${moduleName}...`);
    const module = await loader();
    this.loadedModules.add(moduleName);
    return module;
  }

  isLoaded(moduleName: string): boolean {
    return this.loadedModules.has(moduleName);
  }
}

/**
 * Change detection optimizer
 */
export class ChangeDetectionOptimizer {
  /**
   * Detach change detection for static components
   */
  static detach(cdr: ChangeDetectorRef): void {
    cdr.detach();
  }

  /**
   * Manually trigger change detection
   */
  static detectChanges(cdr: ChangeDetectorRef): void {
    cdr.detectChanges();
  }

  /**
   * Reattach change detection
   */
  static reattach(cdr: ChangeDetectorRef): void {
    cdr.reattach();
  }
}

/**
 * Web Worker pool for CPU-intensive tasks
 */
export class WorkerPool {
  private workers: Worker[] = [];
  private availableWorkers: Worker[] = [];
  private taskQueue: Array<{ task: any; resolve: Function; reject: Function }> = [];

  constructor(workerScript: string, poolSize: number = 4) {
    for (let i = 0; i < poolSize; i++) {
      const worker = new Worker(workerScript);
      this.workers.push(worker);
      this.availableWorkers.push(worker);
    }
  }

  async execute<T>(task: any): Promise<T> {
    return new Promise((resolve, reject) => {
      if (this.availableWorkers.length > 0) {
        const worker = this.availableWorkers.pop()!;
        this.runTask(worker, task, resolve, reject);
      } else {
        this.taskQueue.push({ task, resolve, reject });
      }
    });
  }

  private runTask(worker: Worker, task: any, resolve: Function, reject: Function): void {
    const handleMessage = (event: MessageEvent) => {
      worker.removeEventListener('message', handleMessage);
      worker.removeEventListener('error', handleError);
      this.availableWorkers.push(worker);
      this.processQueue();
      resolve(event.data);
    };

    const handleError = (error: ErrorEvent) => {
      worker.removeEventListener('message', handleMessage);
      worker.removeEventListener('error', handleError);
      this.availableWorkers.push(worker);
      this.processQueue();
      reject(error);
    };

    worker.addEventListener('message', handleMessage);
    worker.addEventListener('error', handleError);
    worker.postMessage(task);
  }

  private processQueue(): void {
    if (this.taskQueue.length > 0 && this.availableWorkers.length > 0) {
      const { task, resolve, reject } = this.taskQueue.shift()!;
      const worker = this.availableWorkers.pop()!;
      this.runTask(worker, task, resolve, reject);
    }
  }

  terminate(): void {
    this.workers.forEach(worker => worker.terminate());
    this.workers = [];
    this.availableWorkers = [];
  }
}

/**
 * Performance monitor
 */
export class PerformanceMonitor {
  private marks = new Map<string, number>();
  private measures: Array<{ name: string; duration: number }> = [];

  mark(name: string): void {
    this.marks.set(name, performance.now());
  }

  measure(name: string, startMark: string, endMark?: string): number {
    const start = this.marks.get(startMark);
    if (!start) {
      console.warn(`Start mark ${startMark} not found`);
      return 0;
    }

    const end = endMark ? this.marks.get(endMark) : performance.now();
    if (!end) {
      console.warn(`End mark ${endMark} not found`);
      return 0;
    }

    const duration = end - start;
    this.measures.push({ name, duration });
    return duration;
  }

  getReport(): { name: string; duration: number }[] {
    return [...this.measures];
  }

  clear(): void {
    this.marks.clear();
    this.measures = [];
  }
}

/**
 * Image lazy loading
 */
export class ImageLazyLoader {
  private observer: IntersectionObserver;

  constructor() {
    this.observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const img = entry.target as HTMLImageElement;
            const src = img.dataset['src'];
            if (src) {
              img.src = src;
              this.observer.unobserve(img);
            }
          }
        });
      },
      {
        rootMargin: '50px',
      }
    );
  }

  observe(element: HTMLImageElement): void {
    this.observer.observe(element);
  }

  disconnect(): void {
    this.observer.disconnect();
  }
}
