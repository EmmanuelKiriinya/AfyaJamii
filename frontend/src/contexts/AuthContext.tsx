import React, { createContext, useContext, useState, useEffect } from 'react';

interface AuthContextType {
  token: string | null;
  username: string | null;
  accountType: string | null;
  login: (token: string, username: string, accountType: string) => void;
  logout: () => void;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [token, setToken] = useState<string | null>(null);
  const [username, setUsername] = useState<string | null>(null);
  const [accountType, setAccountType] = useState<string | null>(null);

  useEffect(() => {
    const storedToken = localStorage.getItem('afyajamii_token');
    const storedUsername = localStorage.getItem('afyajamii_username');
    const storedAccountType = localStorage.getItem('afyajamii_account_type');
    if (storedToken && storedUsername) {
      setToken(storedToken);
      setUsername(storedUsername);
      setAccountType(storedAccountType);
    }
  }, []);

  const login = (newToken: string, newUsername: string, newAccountType: string) => {
    setToken(newToken);
    setUsername(newUsername);
    setAccountType(newAccountType);
    localStorage.setItem('afyajamii_token', newToken);
    localStorage.setItem('afyajamii_username', newUsername);
    localStorage.setItem('afyajamii_account_type', newAccountType);
  };

  const logout = () => {
    setToken(null);
    setUsername(null);
    setAccountType(null);
    localStorage.removeItem('afyajamii_token');
    localStorage.removeItem('afyajamii_username');
    localStorage.removeItem('afyajamii_account_type');
  };

  return (
    <AuthContext.Provider value={{ token, username, accountType, login, logout, isAuthenticated: !!token }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
